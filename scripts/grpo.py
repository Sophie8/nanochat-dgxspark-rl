"""
GRPO (Group Relative Policy Optimization) with LoRA for HuggingFace models in nanochat.

Built on nanochat's infrastructure (compute_init, HFModelWrapper, HFTokenizerWrapper,
DistAdamW, DummyWandb), with GRPO algorithm referenced from TRL.

Key design decisions:
- LoRA for parameter-efficient training (~0.5% of params trainable)
- Base model serves as reference model for KL penalty (no extra memory!)
  -> Just use `with model.disable_lora():` to get base model log probs
- DistAdamW handles gradient sync (nanochat pattern, not DDP)
- Follows chat_rl.py's rollout/training loop structure

Algorithm:
For each optimization step:
  1. Sample prompts from the task/dataset (data parallelism across ranks)
  2. For each prompt, generate G completions using the LoRA policy model
  3. Score each completion using a reward function
  4. Compute group-relative advantages (zscore or mean_only)
  5. Compute per-token log probs under policy (LoRA enabled)
  6. Optionally compute ref log probs (LoRA disabled) for KL penalty
  7. Policy gradient loss with optional PPO-style clipping
  8. Backward + optimizer step (only LoRA params updated, gradient sync via DistAdamW)

Launch:
  Single GPU:
    python -m scripts.qwen3_grpo -- --model-path /path/to/qwen3

  Two DGX Spark nodes (1 GPU each, 200G network):
    # On master:
    MASTER_ADDR=$IP NODE_RANK=0 bash scripts/run_grpo_2node.sh
    # On worker:
    MASTER_ADDR=$IP NODE_RANK=1 bash scripts/run_grpo_2node.sh
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import re
import copy
import math
import itertools
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import (
    compute_init, compute_cleanup, print0, get_base_dir,
    DummyWandb, autodetect_device_type, print_banner,
)
from nanochat.hf_model_wrapper import HFModelWrapper
from nanochat.hf_tokenizer_wrapper import HFTokenizerWrapper, NANOCHAT_SPECIAL_TOKENS

# ---------------------------------------------------------------------------
# SwanLab (graceful fallback, supplements wandb)
# ---------------------------------------------------------------------------
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

# ---------------------------------------------------------------------------
# GSM8K-style reward regex
# ---------------------------------------------------------------------------
_GSM_RE = re.compile(r"####\s*(\-?[0-9\.\,]+)")


# ===========================================================================
# Custom RL Task (local JSONL)
# ===========================================================================
class CustomRLTask:
    """
    RL task from a local JSONL file. Each line must have:
      - "messages": list of chat messages (at least one user message)
      - "answer": ground truth answer string

    Reward: 1.0 if model output contains "#### <correct_answer>", else 0.0
    Compatible with nanochat's task interface (used by chat_rl.py).
    """

    def __init__(self, filepath):
        assert os.path.exists(filepath), f"File not found: {filepath}"
        self.data = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "messages" in obj and "answer" in obj
                self.data.append(obj)
        print0(f"Loaded {len(self.data)} examples from {filepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return item dict (with 'messages' and 'answer')."""
        return self.data[idx]

    def reward(self, item, assistant_response):
        """
        Reward: 1.0 if the correct answer appears in the response.
        Checks multiple formats:
        1. "#### <number>" (GSM8K standard)
        2. "the answer is <number>"
        3. "= <number>" at end of line
        """
        gt = str(item["answer"]).replace(",", "").strip()

        # Format 1: #### <number>
        match = _GSM_RE.search(assistant_response)
        if match:
            pred = match.group(1).replace(",", "").strip()
            if pred == gt:
                return 1.0

        # Format 2: "the answer is <number>" (case insensitive)
        answer_match = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?(\-?[0-9\.\,]+)",
            assistant_response, re.IGNORECASE
        )
        if answer_match:
            pred = answer_match.group(1).replace(",", "").replace("$", "").strip()
            if pred == gt:
                return 1.0

        # Format 3: boxed answer \boxed{<number>}
        boxed_match = re.search(r"\\boxed\{(\-?[0-9\.\,]+)\}", assistant_response)
        if boxed_match:
            pred = boxed_match.group(1).replace(",", "").strip()
            if pred == gt:
                return 1.0

        return 0.0

    def evaluate(self, item, assistant_response):
        return int(self.reward(item, assistant_response))


# ===========================================================================
# CLI arguments (follows chat_rl.py pattern)
# ===========================================================================
print_banner()

parser = argparse.ArgumentParser(description="GRPO with LoRA for HuggingFace models (e.g. Qwen3)")
# Model
parser.add_argument("--model-path", type=str, required=True,
                    help="HuggingFace model name or local path (e.g. Qwen/Qwen3-8B)")
# Logging
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    help="float32|bfloat16")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1)
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=2,
                    help="max batch size per forward pass (reduce if OOM)")
parser.add_argument("--examples-per-step", type=int, default=2,
                    help="total examples (prompts) per optimization step across all ranks")
parser.add_argument("--num-generations", type=int, default=4,
                    help="number of completions per prompt (G in GRPO)")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top-k", type=int, default=50)
# GRPO specific
parser.add_argument("--advantage-norm", type=str, default="zscore",
                    choices=["zscore", "mean_only"])
parser.add_argument("--beta", type=float, default=0.01,
                    help="KL penalty coefficient (uses base model via disable_lora, no extra memory)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO-style clipping epsilon (0 = no clipping)")
# LoRA parameters
parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--lora-dropout", type=float, default=0.05)
parser.add_argument("--lora-target", type=str, nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])
# Optimization
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--warmup-ratio", type=float, default=0.1)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--init-lr-frac", type=float, default=0.1,
                    help="initial LR as fraction of base LR (for warmup)")
# Task
parser.add_argument("--task", type=str, default="data/gsm8k_mini_test.jsonl",
                    help="Path to a local JSONL file with 'messages' and 'answer' fields")
parser.add_argument("--val-data", type=str, default=None)
# Chat format
parser.add_argument("--chat-format", type=str, default="native",
                    choices=["native", "nanochat"],
                    help="prompt format: 'native' uses model's chat template, "
                         "'nanochat' uses nanochat's special tokens")
# Evaluation
parser.add_argument("--eval-every", type=int, default=30)
parser.add_argument("--eval-examples", type=int, default=50)
parser.add_argument("--eval-samples", type=int, default=1)
# Checkpointing
parser.add_argument("--save-every", type=int, default=60)
parser.add_argument("--output-dir", type=str, default=None)
# SwanLab
parser.add_argument("--swanlab-mode", type=str, default="disabled",
                    choices=["cloud", "local", "disabled"])
parser.add_argument("--run-name", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()


# ===========================================================================
# Compute init (nanochat standard)
# ===========================================================================
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging (nanochat standard)
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-grpo", name=args.run, config=user_config)

# SwanLab (supplemental)
swanlab_run = None
if master_process and HAS_SWANLAB and args.swanlab_mode != "disabled":
    try:
        output_dir = args.output_dir or os.path.join(get_base_dir(), "grpo_checkpoints")
        swanlab_run = swanlab.init(
            project="nanochat-grpo",
            experiment_name=args.run_name or args.run,
            config=user_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(output_dir, "swanlog"),
        )
        print0(f"SwanLab initialized ({args.swanlab_mode})")
    except Exception as e:
        print0(f"SwanLab init failed: {e}")


def log_all(metrics, step=None):
    """Log to both wandb and swanlab."""
    wandb_run.log({"step": step, **metrics} if step is not None else metrics)
    if swanlab_run is not None:
        try:
            swanlab.log(metrics, step=step)
        except Exception:
            pass


# ===========================================================================
# Load tokenizer + model with LoRA
# ===========================================================================
print0("=" * 80)
print0(f"Loading tokenizer from {args.model_path}...")
tokenizer = HFTokenizerWrapper(args.model_path)

print0(f"Loading policy model from {args.model_path}...")
model = HFModelWrapper(
    args.model_path,
    device=device,
    dtype=ptdtype,
    extra_special_tokens=NANOCHAT_SPECIAL_TOKENS if tokenizer.num_added_tokens > 0 else None,
)
model.sync_vocab_size(tokenizer.get_vocab_size())

# Apply LoRA
print0(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
model.apply_lora(
    r=args.lora_r,
    alpha=args.lora_alpha,
    dropout=args.lora_dropout,
    target_modules=args.lora_target,
)


# ===========================================================================
# Load task
# ===========================================================================
assert os.path.isfile(args.task), f"Task file not found: {args.task}"
train_task = CustomRLTask(args.task)
val_task = CustomRLTask(args.val_data) if args.val_data else None

num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Task: {args.task}, Train examples: {len(train_task)}, Steps: {num_steps}")


# ===========================================================================
# Setup optimizer (DistAdamW for distributed, AdamW for single — nanochat pattern)
# ===========================================================================
optimizer = model.setup_optimizer(lr=args.lr, weight_decay=args.weight_decay)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# LR scheduler: linear rampdown to zero (same as chat_rl.py)
# The initial_lr is already set to base_lr * init_lr_frac
# The multiplier ramps from 1.0 to 0.0 over num_steps
def get_lr_multiplier(it):
    return max(0.0, 1.0 - it / max(num_steps, 1))

# Data parallelism setup
assert args.examples_per_step % ddp_world_size == 0, \
    f"examples_per_step ({args.examples_per_step}) must be divisible by world_size ({ddp_world_size})"
examples_per_rank = args.examples_per_step // ddp_world_size

# Output directory
if args.output_dir is None:
    model_short_name = args.model_path.rstrip("/").split("/")[-1]
    args.output_dir = os.path.join(get_base_dir(), "grpo_checkpoints", model_short_name)
print0(f"Output directory: {args.output_dir}")


# ===========================================================================
# Prompt preparation
# ===========================================================================
# System prompt for math tasks — instructs the model to use "#### N" answer format
MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "At the end of your solution, provide the final numerical answer on a new line "
    "in this exact format: #### <number>\n"
    "For example: #### 42"
)

def prepare_prompt(item):
    """
    Prepare prompt tokens for generation.
    - native: use HF chat template (best for instruct models)
    - nanochat: use nanochat special tokens (for nanochat-finetuned models)

    Adds a system prompt for math tasks to guide the model to produce
    the expected "#### <answer>" format needed by the reward function.
    """
    if args.chat_format == "nanochat":
        conversation = {"messages": item["messages"]}
        return tokenizer.render_for_completion(conversation)
    else:
        messages = []
        has_system = False
        for msg in item["messages"]:
            if msg["role"] == "system":
                messages.append(msg)
                has_system = True
            elif msg["role"] == "user":
                messages.append(msg)
            elif msg["role"] == "assistant":
                break  # stop before ground truth
        # Add math system prompt if no system message exists
        if not has_system:
            messages.insert(0, {"role": "system", "content": MATH_SYSTEM_PROMPT})
        prompt_text = tokenizer.hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.hf_tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_ids


# ===========================================================================
# Generation (uses HFModelWrapper.generate())
# ===========================================================================
@torch.no_grad()
def generate_completions(prompt_tokens, num_samples, max_new_tokens,
                         temperature, top_k, seed):
    """
    Generate multiple completions for a single prompt.
    Processes in mini-batches of device_batch_size to prevent OOM.
    Returns: list of list[int] (completion token ids, prompt stripped)
    """
    model.eval()
    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    all_completion_ids = []
    num_batches = (num_samples + args.device_batch_size - 1) // args.device_batch_size

    for b in range(num_batches):
        batch_size = min(args.device_batch_size, num_samples - b * args.device_batch_size)
        batch_prompt_ids = prompt_ids.expand(batch_size, -1)

        torch.manual_seed(seed + b * 1000)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed + b * 1000)

        with autocast_ctx:
            output_ids = model.generate(
                batch_prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                do_sample=True,
                pad_token_id=tokenizer.hf_tokenizer.pad_token_id,
                eos_token_id=tokenizer.hf_tokenizer.eos_token_id,
            )

        for i in range(batch_size):
            comp_ids = output_ids[i, len(prompt_tokens):].tolist()
            eos_id = tokenizer.hf_tokenizer.eos_token_id
            if eos_id is not None and eos_id in comp_ids:
                comp_ids = comp_ids[:comp_ids.index(eos_id) + 1]
            all_completion_ids.append(comp_ids)

    return all_completion_ids


# ===========================================================================
# Rollout / sampling generator (follows chat_rl.py pattern)
# ===========================================================================
step = 0  # module-level step counter

@torch.no_grad()
def get_batch():
    """
    Generator yielding GRPO training batches (one prompt + G completions each).
    Each rank cycles through different examples for data parallelism.

    Yields:
        completion_ids_list, inputs, targets, completion_mask, rewards, advantages
    """
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

    for example_idx in itertools.cycle(rank_indices):
        item = train_task[example_idx]
        prompt_tokens = prepare_prompt(item)
        prefix_length = len(prompt_tokens)

        # Generate G completions
        completion_ids_list = generate_completions(
            prompt_tokens,
            num_samples=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=hash((step, example_idx)) & 0x7FFFFFFF,
        )

        # Compute rewards
        rewards = []
        for comp_ids in completion_ids_list:
            comp_text = tokenizer.decode(comp_ids)
            reward = train_task.reward(item, comp_text)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        # GRPO: group-relative advantages
        mu = rewards.mean()
        if args.advantage_norm == "zscore":
            sigma = rewards.std().clamp(min=1e-8)
            advantages = (rewards - mu) / sigma
        else:
            advantages = rewards - mu

        # Build padded sequences
        pad_id = tokenizer.hf_tokenizer.pad_token_id or 0
        all_sequences = [prompt_tokens + comp_ids for comp_ids in completion_ids_list]
        max_length = max(len(seq) for seq in all_sequences)

        padded_sequences = []
        masks = []
        for seq in all_sequences:
            pad_len = max_length - len(seq)
            padded_sequences.append(seq + [pad_id] * pad_len)
            masks.append([0] * prefix_length + [1] * (len(seq) - prefix_length) + [0] * pad_len)

        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(masks, dtype=torch.long, device=device)

        # Autoregressive shift (same as chat_rl.py)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_tensor[:, 1:] == 0] = -1
        completion_mask = mask_tensor[:, 1:].float()

        yield completion_ids_list, inputs, targets, completion_mask, rewards, advantages


# ===========================================================================
# Evaluation (follows chat_rl.py's run_gsm8k_eval pattern)
# ===========================================================================
@torch.no_grad()
def run_eval(task, num_examples, num_samples=1,
             max_new_tokens=512, temperature=0.0, top_k=50):
    """Evaluate pass@k on a task. Returns (pass_at_k_tensor, total_evaluated)."""
    model.eval()
    max_examples = min(num_examples, len(task))
    pass_at_k = torch.zeros(num_samples, device=device)
    num_evaluated = 0

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        item = task[idx]
        prompt_tokens = prepare_prompt(item)

        completions = generate_completions(
            prompt_tokens, num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature if num_samples > 1 else 0.0,
            top_k=top_k,
            seed=hash(("eval", idx)) & 0x7FFFFFFF,
        )

        outcomes = []
        for comp_ids in completions:
            comp_text = tokenizer.decode(comp_ids)
            is_correct = task.evaluate(item, comp_text)
            outcomes.append(is_correct)

        for k in range(1, num_samples + 1):
            if any(outcomes[:k]):
                pass_at_k[k - 1] += 1
        num_evaluated += 1

    # Aggregate across ranks
    num_evaluated_tensor = torch.tensor(num_evaluated, dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(num_evaluated_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(pass_at_k, op=dist.ReduceOp.SUM)
    total_evaluated = num_evaluated_tensor.item()
    if total_evaluated > 0:
        pass_at_k = pass_at_k / total_evaluated
    return pass_at_k, total_evaluated


# ===========================================================================
# Training loop (follows chat_rl.py structure)
# ===========================================================================
print0("=" * 80)
print0(f"GRPO Training with LoRA")
print0(f"  Model: {args.model_path}")
print0(f"  Task: {args.task} ({len(train_task)} examples)")
print0(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha} (scaling={args.lora_alpha/args.lora_r:.1f}x)")
print0(f"  Generations/prompt (G): {args.num_generations}")
print0(f"  Examples/step: {args.examples_per_step} ({examples_per_rank}/rank)")
print0(f"  Sequences/step: {args.examples_per_step * args.num_generations}")
print0(f"  Advantage norm: {args.advantage_norm}")
print0(f"  KL beta: {args.beta} {'(base model as ref, no extra mem)' if args.beta > 0 else '(no KL)'}")
print0(f"  Clip eps: {args.clip_eps} {'' if args.clip_eps > 0 else '(no clipping)'}")
print0(f"  LR: {args.lr} (linear rampdown, init_frac={args.init_lr_frac})")
print0(f"  Chat format: {args.chat_format}")
print0(f"  Generation: temp={args.temperature}, top_k={args.top_k}, max_tokens={args.max_new_tokens}")
print0(f"  Steps: {num_steps}, World size: {ddp_world_size}")
print0("=" * 80)

batch_iterator = get_batch()
total_training_time = 0
smooth_reward = 0
ema_beta = 0.9

for step in range(num_steps):

    # ---- Evaluate ----
    if val_task is not None and step % args.eval_every == 0:
        with autocast_ctx:
            pass_at_k, num_eval = run_eval(
                val_task, num_examples=args.eval_examples,
                num_samples=args.eval_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0 if args.eval_samples > 1 else 0.0,
                top_k=args.top_k,
            )
        print_passk = [f"Pass@{k}: {pass_at_k[k-1].item():.4f}" for k in range(1, args.eval_samples + 1)]
        print0(f"Step {step} | Eval ({num_eval} examples) | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": pass_at_k[k-1].item() for k in range(1, args.eval_samples + 1)}
        log_all(log_passk, step=step)

    # ---- Forward / backward on rollouts ----
    synchronize()
    t0 = time.time()

    rewards_list = []
    sequence_lengths = []

    for example_step in range(examples_per_rank):
        completions, inputs, targets, comp_mask, rewards, advantages = next(batch_iterator)

        model.train()

        # Process in mini-batches
        B = inputs.size(0)
        num_passes = max(1, (B + args.device_batch_size - 1) // args.device_batch_size)

        for pass_idx in range(num_passes):
            b0 = pass_idx * args.device_batch_size
            b1 = min((pass_idx + 1) * args.device_batch_size, B)

            inp = inputs[b0:b1]
            tgt = targets[b0:b1]
            mask = comp_mask[b0:b1]
            adv = advantages[b0:b1]

            with autocast_ctx:
                # Per-token log probs from policy (LoRA enabled)
                logp = -model(inp, tgt, loss_reduction='none').view(inp.size(0), inp.size(1))

                # Per-token log probs from reference (LoRA disabled)
                ref_logp = None
                if args.beta > 0:
                    with model.disable_lora():
                        ref_logp = -model(inp, tgt, loss_reduction='none').view(inp.size(0), inp.size(1))

            # ---- GRPO Policy Gradient ----
            if args.clip_eps > 0 and ref_logp is not None:
                # PPO-style clipping (TRL GRPO mode)
                ratio = torch.exp(logp - ref_logp.detach())
                clipped_ratio = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_obj = (torch.min(
                    ratio * adv.unsqueeze(-1),
                    clipped_ratio * adv.unsqueeze(-1)
                ) * mask).sum()
            else:
                # Simple REINFORCE (nanochat-style, no clipping)
                pg_obj = (logp * mask * adv.unsqueeze(-1)).sum()

            num_valid = mask.sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

            loss = -pg_obj

            # KL penalty
            if ref_logp is not None and args.beta > 0:
                kl = ((logp - ref_logp.detach()) * mask).sum() / num_valid
                loss = loss + args.beta * kl

            loss.backward()
            print0(f"  step {step}/{num_steps} | ex {example_step} | pass {pass_idx} | "
                   f"loss: {loss.item():.6f} | avg reward: {rewards.mean().item():.3f}")

        rewards_list.append(rewards.mean().item())
        sequence_lengths.extend(len(comp) for comp in completions)

        # Print a sample completion for debugging (first example of first step only)
        if step < 2 and example_step == 0 and master_process:
            sample_text = tokenizer.decode(completions[0][:200])  # first 200 tokens
            print0(f"   Sample completion (step={step}, reward={rewards[0].item():.1f}):")
            print0(f"     {sample_text[:300]}{'...' if len(sample_text) > 300 else ''}")

    # ---- Aggregate logging across ranks ----
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_seq_len = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0
    if ddp:
        mean_reward_t = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_seq_len_t = torch.tensor(mean_seq_len, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_seq_len_t, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_t.item()
        mean_seq_len = mean_seq_len_t.item()

    # ---- Gradient clipping + optimizer step ----
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
    ).item()

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 5:
        total_training_time += dt

    # ---- Logging ----
    smooth_reward = ema_beta * smooth_reward + (1 - ema_beta) * mean_reward
    debiased_reward = smooth_reward / (1 - ema_beta ** (step + 1))

    print0(f"Step {step}/{num_steps} | reward: {debiased_reward:.4f} | "
           f"mean_seq_len: {mean_seq_len:.1f} | lrm: {lrm:.3f} | "
           f"grad_norm: {grad_norm:.3f} | dt: {dt:.2f}s | total: {total_training_time/60:.1f}m")

    log_all({
        "reward": debiased_reward,
        "raw_reward": mean_reward,
        "sequence_length": mean_seq_len,
        "lrm": lrm,
        "grad_norm": grad_norm,
        "dt": dt,
    }, step=step)

    # ---- Save checkpoint (LoRA adapter only) ----
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        save_dir = os.path.join(args.output_dir, f"step_{step:06d}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_hf(save_dir)  # saves LoRA adapter (since LoRA is active)
        tokenizer.save(save_dir)
        meta = {
            "step": step,
            "reward": debiased_reward,
            "task": args.task,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "config": user_config,
        }
        with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print0(f"[OK] Saved LoRA checkpoint to {save_dir}")

# Close the generator to avoid cleanup warnings
batch_iterator.close()

# Final stats
print0("=" * 80)
print0(f"GRPO Training complete!")
print0(f"  Final reward: {debiased_reward:.4f}")
print0(f"  Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"  Total training time: {total_training_time / 60:.2f}m")
print0(f"  LoRA adapter saved to: {args.output_dir}")
print0("=" * 80)

# Cleanup
wandb_run.finish()
if swanlab_run is not None:
    try:
        swanlab.finish()
    except Exception:
        pass
compute_cleanup()
