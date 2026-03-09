"""
Search-R1 GRPO Training with LoRA
支持技能学习、工具调用和多轮训练

Launch:
  Single GPU:
    python -m scripts.search_r1_grpo --model-path Qwen/Qwen3-8B

  Distributed (2 nodes):
    # Node 0:
    MASTER_ADDR=<ip> NODE_RANK=0 bash scripts/run_search_r1_2node.sh
    # Node 1:
    MASTER_ADDR=<ip> NODE_RANK=1 bash scripts/run_search_r1_2node.sh
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import random
import itertools
import swanlab
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from nanochat.common import (
    compute_init, compute_cleanup, print0, get_base_dir,
    autodetect_device_type, print_banner,
)
from nanochat.hf_model_wrapper import HFModelWrapper
from nanochat.hf_tokenizer_wrapper import HFTokenizerWrapper, NANOCHAT_SPECIAL_TOKENS

from tasks.search_r1 import SearchR1Task, generate_search_r1_dataset
from tools.search_tools import SEARCH_TOOL_DEFINITION

print_banner()

# ===============================================================================
# CLI arguments
# ===============================================================================
parser = argparse.ArgumentParser(description="Search-R1 GRPO Training with LoRA")
# Model
parser.add_argument("--model-path", type=str, required=True,
                    help="HuggingFace model name or path")
# Logging
parser.add_argument("--run", type=str, default="search-r1-grpo",
                    help="SwanLab experiment name")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16")
# Training
parser.add_argument("--num-epochs", type=int, default=3,
                    help="number of training epochs (supports multi-epoch)")
parser.add_argument("--examples-per-step", type=int, default=4,
                    help="prompts per optimization step across all ranks")
parser.add_argument("--num-generations", type=int, default=8,
                    help="completions per prompt (G in GRPO)")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top-k", type=int, default=50)
# GRPO
parser.add_argument("--device-batch-size", type=int, default=2,
                    help="max batch size per forward pass")
parser.add_argument("--advantage-norm", type=str, default="zscore",
                    choices=["zscore", "mean_only"])
parser.add_argument("--beta", type=float, default=0.01,
                    help="KL penalty coefficient")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO-style clipping epsilon")
# LoRA
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.05)
parser.add_argument("--lora-target", type=str, nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])
# Optimization
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--warmup-ratio", type=float, default=0.1)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
# Data
parser.add_argument("--train-data", type=str, default=None,
                    help="training data path (JSONL)")
parser.add_argument("--val-data", type=str, default=None,
                    help="validation data path (JSONL)")
parser.add_argument("--search-engine", type=str, default="duckduckgo",
                    choices=["gemini", "tavily", "serper", "duckduckgo", "mock"],
                    help="search engine for tool calls (default: duckduckgo)")
parser.add_argument("--search-proxy", type=str, default=None,
                    help="HTTP proxy for search engine (e.g. http://<PROXY_IP>:7890)")
parser.add_argument("--enable-memory", action="store_true",
                    help="enable conversation memory for multi-turn training")
parser.add_argument("--max-search-turns", type=int, default=5,
                    help="max search-inject turns per rollout completion")
# Evaluation
parser.add_argument("--eval-every", type=int, default=30)
parser.add_argument("--eval-examples", type=int, default=50)
# Output
parser.add_argument("--save-every", type=int, default=60)
parser.add_argument("--output-dir", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()

# ===============================================================================
# Compute init
# ===============================================================================
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
# Print compute initialization info
print0("=" * 80)
print0("Compute Initialization")
print0("=" * 80)
print0(f"Device type: {device_type}")
print0(f"Device: {device}")
print0(f"DDP enabled: {ddp}")
if ddp:
    print0(f"DDP rank: {ddp_rank}/{ddp_world_size}")
    print0(f"DDP local rank: {ddp_local_rank}")
print0(f"Master process: {master_process}")
print0(f"Precision setup: {ptdtype}")
print0(f"Autocast enabled: {device_type == 'cuda'}")
print0(f"CUDA synchronization: {device_type == 'cuda'}")
print0(f"Memory tracking: {device_type == 'cuda'}")
print0("=" * 80)

# SwanLab logging
if master_process and args.run != "dummy":
    swanlab.init(
        project="nanochat-search-r1",
        experiment_name=args.run,
        config=user_config,
    )
    use_swanlab = True
else:
    use_swanlab = False

# ===============================================================================
# Load tokenizer + model with LoRA
# ===============================================================================
print0("=" * 80)
print0(f"Loading tokenizer from {args.model_path}...")
tokenizer = HFTokenizerWrapper(args.model_path)

print0(f"Loading model from {args.model_path}...")
model = HFModelWrapper(
    args.model_path,
    device=device,
    dtype=ptdtype,
    extra_special_tokens=NANOCHAT_SPECIAL_TOKENS if tokenizer.num_added_tokens > 0 else None,
)
model.sync_vocab_size(tokenizer.get_vocab_size())

print0(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
model.apply_lora(
    r=args.lora_r,
    alpha=args.lora_alpha,
    dropout=args.lora_dropout,
    target_modules=args.lora_target,
)

# ===============================================================================
# Load task
# ===============================================================================
# 设置搜索代理（仅供搜索引擎使用，不设全局代理以免影响 SwanLab 等服务）
if args.search_proxy:
    os.environ["SEARCH_PROXY"] = args.search_proxy
    print0(f"Search proxy (search-only): {args.search_proxy}")

base_dir = get_base_dir()

# 生成或加载训练数据
if args.train_data is None:
    args.train_data = os.path.join(base_dir, "data", "search_r1_train.jsonl")
    if not os.path.exists(args.train_data):
        os.makedirs(os.path.dirname(args.train_data), exist_ok=True)
        print0(f"Generating training data to {args.train_data}...")
        generate_search_r1_dataset(args.train_data, num_examples=200)

if args.val_data is None:
    args.val_data = os.path.join(base_dir, "data", "search_r1_val.jsonl")
    if not os.path.exists(args.val_data):
        os.makedirs(os.path.dirname(args.val_data), exist_ok=True)
        print0(f"Generating validation data to {args.val_data}...")
        generate_search_r1_dataset(args.val_data, num_examples=50)

train_task = SearchR1Task(
    args.train_data,
    split="train",
    search_engine=args.search_engine,
    enable_memory=args.enable_memory
)
val_task = SearchR1Task(
    args.val_data,
    split="test",
    search_engine=args.search_engine,
    enable_memory=False  # 评估时不需要记忆
)

num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Training: {len(train_task)} examples, {num_steps} steps ({args.num_epochs} epochs)")
print0(f"Validation: {len(val_task)} examples")

# ===============================================================================
# Setup optimizer
# ===============================================================================
optimizer = model.setup_optimizer(lr=args.lr, weight_decay=args.weight_decay)

# LR scheduler: warmup + linear decay
warmup_steps = int(num_steps * args.warmup_ratio)
def get_lr_multiplier(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    else:
        return max(0.0, 1.0 - (step - warmup_steps) / max(num_steps - warmup_steps, 1))

# Data parallelism
assert args.examples_per_step % ddp_world_size == 0
examples_per_rank = args.examples_per_step // ddp_world_size

# Output directory
if args.output_dir is None:
    model_short_name = args.model_path.rstrip("/").split("/")[-1]
    args.output_dir = os.path.join(base_dir, "search_r1_checkpoints", model_short_name)
print0(f"Output directory: {args.output_dir}")

# ===============================================================================
# System prompt with tool definitions
# ===============================================================================
SYSTEM_PROMPT_WITH_TOOLS = """\
你是一个有用的AI助手，可以使用网络搜索工具来获取信息并进行多步推理。

可用工具：
- web_search：搜索网络获取最新信息

回答问题时，请遵循以下流程：
1. 在 <think>...</think> 标签中进行推理思考
2. 需要搜索时，使用 <search>你的搜索查询</search> 标签
3. 搜索结果会以 <information>...</information> 标签返回给你
4. 根据搜索结果继续推理，如果信息不够可以再次搜索
5. 最终在推理完成后直接给出答案

示例格式：
<think>我需要先了解X，然后再查Y</think>
<search>X相关查询</search>
<information>搜索结果...</information>
<think>根据搜索结果，我了解到...但还需要进一步了解Y</think>
<search>Y相关查询</search>
<information>搜索结果...</information>
<think>综合以上信息，我可以得出结论</think>
最终答案"""

def prepare_prompt(item):
    """准备prompt（添加system和工具定义）"""
    messages = item["messages"].copy()
    
    # 替换 system prompt
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = SYSTEM_PROMPT_WITH_TOOLS
    else:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOLS})
    
    # 使用 HF chat template
    prompt_text = tokenizer.hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.hf_tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_ids

# ===============================================================================
# Generation (multi-turn rollout support)
# ===============================================================================
SEARCH_TAG_END = "</search>"
INFO_TAG_START = "<information>"
INFO_TAG_END = "</information>"
MAX_SEARCH_TURNS = args.max_search_turns  # 每个 completion 最多执行几轮搜索


@torch.no_grad()
def generate_single_segment(context_ids, max_new_tokens, temperature, top_k, seed):
    """从给定 context 生成一段文本（单次生成，不处理搜索标签）"""
    model.eval()
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    with autocast_ctx:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.hf_tokenizer.pad_token_id,
            eos_token_id=tokenizer.hf_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, len(context_ids):].tolist()
    eos_id = tokenizer.hf_tokenizer.eos_token_id
    if eos_id is not None and eos_id in new_ids:
        new_ids = new_ids[:new_ids.index(eos_id) + 1]
    return new_ids


@torch.no_grad()
def generate_multiturn_completion(
    prompt_tokens, max_new_tokens, temperature, top_k, seed
):
    """
    多轮 rollout 生成：
      1. 模型从 prompt 生成
      2. 如果输出包含 </search>，截断 -> 执行搜索 -> 注入 <information>
      3. 从扩展后的 context 继续生成
      4. 重复直到无搜索或达到 MAX_SEARCH_TURNS

    返回:
      - full_completion_ids: 完整响应的 token ids（仅 completion 部分）
      - full_text: 完整响应文本
      - search_mask: 标记哪些 token 是搜索结果（用于 loss mask）
    """
    context_ids = list(prompt_tokens)
    all_new_ids = []         # 收集所有生成的 token ids
    search_result_ranges = []  # [(start, end)] 搜索结果在 all_new_ids 中的位置

    tokens_remaining = max_new_tokens

    for turn in range(MAX_SEARCH_TURNS):
        if tokens_remaining <= 0:
            break

        # 生成一段
        seg_ids = generate_single_segment(
            context_ids, min(tokens_remaining, max_new_tokens // 2),
            temperature, top_k, seed + turn * 1000
        )
        seg_text = tokenizer.decode(seg_ids)

        # 检查是否包含 </search>
        if SEARCH_TAG_END in seg_text:
            # 截断到 </search> 之后
            cut_idx = seg_text.index(SEARCH_TAG_END) + len(SEARCH_TAG_END)
            seg_text_truncated = seg_text[:cut_idx]
            seg_ids_truncated = tokenizer.hf_tokenizer.encode(
                seg_text_truncated, add_special_tokens=False)

            all_new_ids.extend(seg_ids_truncated)
            tokens_remaining -= len(seg_ids_truncated)

            # 提取搜索查询
            queries = train_task.parse_search_tags(seg_text_truncated)
            if queries:
                query = queries[-1]
                search_result = train_task.execute_search(query)
                info_text = f"\n{INFO_TAG_START}{search_result}{INFO_TAG_END}\n"
                info_ids = tokenizer.hf_tokenizer.encode(
                    info_text, add_special_tokens=False)

                # 标记搜索结果范围（用于 loss mask）
                start_pos = len(all_new_ids)
                all_new_ids.extend(info_ids)
                search_result_ranges.append(
                    (start_pos, start_pos + len(info_ids)))

                # 更新 context
                context_ids = list(prompt_tokens) + all_new_ids
                tokens_remaining -= len(info_ids)
            else:
                break
        else:
            # 没有搜索标签 -> 最终段
            all_new_ids.extend(seg_ids)
            tokens_remaining -= len(seg_ids)
            break

    # 构建 search_mask: 1 = 模型生成的 token, 0 = 搜索结果 token（不计入loss）
    search_mask = [1] * len(all_new_ids)
    for start, end in search_result_ranges:
        for i in range(start, min(end, len(search_mask))):
            search_mask[i] = 0

    full_text = tokenizer.decode(all_new_ids)
    return all_new_ids, full_text, search_mask


@torch.no_grad()
def generate_completions(prompt_tokens, num_samples, max_new_tokens,
                         temperature, top_k, seed):
    """生成多个 completions（多轮 rollout 版本）"""
    all_completion_ids = []
    all_texts = []
    all_search_masks = []

    for i in range(num_samples):
        comp_ids, comp_text, s_mask = generate_multiturn_completion(
            prompt_tokens, max_new_tokens, temperature, top_k,
            seed + i * 10000)
        all_completion_ids.append(comp_ids)
        all_texts.append(comp_text)
        all_search_masks.append(s_mask)

    return all_completion_ids, all_texts, all_search_masks

# ===============================================================================
# Rollout generator
# ===============================================================================
step = 0
epoch = 0

@torch.no_grad()
def get_batch():
    """生成GRPO训练批次（多轮 rollout 版本）

    关键改进：
    - 使用 generate_completions 进行多轮 rollout（拦截 search -> 注入 information）
    - 搜索结果 token 通过 search_mask=0 从 loss 中排除（防止模型记忆检索内容）
    """
    global step, epoch

    for current_epoch in range(args.num_epochs):
        epoch = current_epoch

        # 每个 rank 处理不同的样例
        all_indices = list(range(len(train_task)))
        rng = random.Random(42 + current_epoch)
        rng.shuffle(all_indices)
        rank_indices = [idx for i, idx in enumerate(all_indices)
                        if i % ddp_world_size == ddp_rank]

        for example_idx in rank_indices:
            item = train_task[example_idx]
            prompt_tokens = prepare_prompt(item)
            prefix_length = len(prompt_tokens)

            # 多轮 rollout 生成 G 个 completions
            completion_ids_list, completions_text, search_masks = \
                generate_completions(
                    prompt_tokens,
                    num_samples=args.num_generations,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=hash((step, example_idx, epoch)) & 0x7FFFFFFF,
                )

            # 计算奖励
            rewards = []
            for comp_text in completions_text:
                reward = train_task.reward(item, comp_text)
                rewards.append(reward)
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)

            # 更新记忆
            if args.enable_memory:
                best_idx = rewards.argmax().item()
                train_task.update_memory(example_idx, completions_text[best_idx])

            # GRPO: group-relative advantages
            mu = rewards.mean()
            if args.advantage_norm == "zscore":
                sigma = rewards.std().clamp(min=1e-8)
                advantages = (rewards - mu) / sigma
            else:
                advantages = rewards - mu

            # 构建 padded sequences，使用 search_mask 排除检索结果 token
            pad_id = tokenizer.hf_tokenizer.pad_token_id or 0
            all_sequences = [prompt_tokens + cids for cids in completion_ids_list]
            max_length = max(len(seq) for seq in all_sequences)

            padded_sequences = []
            masks = []
            for i, seq in enumerate(all_sequences):
                pad_len = max_length - len(seq)
                padded_sequences.append(seq + [pad_id] * pad_len)

                # mask = 0 for prompt + 搜索结果 + padding
                # mask = 1 for 模型自己生成的 token
                s_mask = search_masks[i]
                comp_mask_part = list(s_mask) + [0] * (
                    len(seq) - prefix_length - len(s_mask))
                full_mask = [0] * prefix_length + comp_mask_part + [0] * pad_len
                # 确保长度匹配
                full_mask = full_mask[:max_length]
                masks.append(full_mask)

            ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
            mask_tensor = torch.tensor(masks, dtype=torch.long, device=device)

            # Autoregressive shift
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_tensor[:, 1:] == 0] = -1
            completion_mask = mask_tensor[:, 1:].float()

            yield (completion_ids_list, inputs, targets, completion_mask,
                   rewards, advantages, completions_text, item)

# ===============================================================================
# Evaluation
# ===============================================================================
@torch.no_grad()
def run_eval(num_examples=50):
    """评估（使用多轮 rollout）"""
    model.eval()
    total_reward = 0.0
    total_correct = 0
    num_evaluated = 0

    for idx in range(ddp_rank, min(num_examples, len(val_task)), ddp_world_size):
        item = val_task[idx]
        prompt_tokens = prepare_prompt(item)

        # 多轮 rollout 贪婪生成
        comp_ids, comp_text, _ = generate_multiturn_completion(
            prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,  # greedy
            top_k=0,
            seed=hash(("eval", idx)) & 0x7FFFFFFF,
        )

        reward = val_task.reward(item, comp_text)
        is_correct = val_task.evaluate(item, comp_text)

        total_reward += reward
        total_correct += is_correct
        num_evaluated += 1
    
    # Aggregate
    reward_tensor = torch.tensor(total_reward, dtype=torch.float, device=device)
    correct_tensor = torch.tensor(total_correct, dtype=torch.float, device=device)
    count_tensor = torch.tensor(num_evaluated, dtype=torch.long, device=device)
    
    if ddp:
        dist.all_reduce(reward_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    
    total_count = count_tensor.item()
    if total_count > 0:
        avg_reward = reward_tensor.item() / total_count
        accuracy = correct_tensor.item() / total_count
    else:
        avg_reward, accuracy = 0.0, 0.0
    
    return avg_reward, accuracy, total_count

# ===============================================================================
# Training loop
# ===============================================================================
print0("=" * 80)
print0(f"Search-R1 GRPO Training (Multi-turn Rollout)")
print0(f"  Model: {args.model_path}")
print0(f"  Data: {args.train_data}")
print0(f"  Search engine: {args.search_engine}")
print0(f"  Memory enabled: {args.enable_memory}")
print0(f"  Max search turns/completion: {MAX_SEARCH_TURNS}")
print0(f"  LoRA: r={args.lora_r}, α={args.lora_alpha}")
print0(f"  Epochs: {args.num_epochs}")
print0(f"  Steps: {num_steps}")
print0(f"  Examples/step: {args.examples_per_step} ({examples_per_rank}/rank)")
print0(f"  Generations/prompt: {args.num_generations}")
print0("=" * 80)

batch_iterator = get_batch()
total_training_time = 0
smooth_reward = 0
ema_beta = 0.9
debiased_reward = 0.0

for step in range(num_steps):
    
    # === Evaluate ===
    if step % args.eval_every == 0:
        with autocast_ctx:
            avg_reward, accuracy, num_eval = run_eval(num_examples=args.eval_examples)
        print0(f"Step {step} | Epoch {epoch} | Eval ({num_eval} examples) | "
               f"Reward: {avg_reward:.4f} | Accuracy: {accuracy:.2%}")
        if use_swanlab:
            swanlab.log({
                "step": step,
                "epoch": epoch,
                "eval/reward": avg_reward,
                "eval/accuracy": accuracy,
            })
    
    # === Training ===
    synchronize()
    t0 = time.time()
    
    rewards_list = []
    
    for _ in range(examples_per_rank):
        completions, inputs, targets, comp_mask, rewards, advantages, texts, item = next(batch_iterator)
        
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
                # Policy log probs
                logp = -model(inp, tgt, loss_reduction='none').view(inp.size(0), inp.size(1))
                
                # Reference log probs (base model via disable_lora)
                ref_logp = None
                if args.beta > 0:
                    with model.disable_lora():
                        ref_logp = -model(inp, tgt, loss_reduction='none').view(inp.size(0), inp.size(1))
            
            # GRPO policy gradient
            if args.clip_eps > 0 and ref_logp is not None:
                ratio = torch.exp(logp - ref_logp.detach())
                clipped_ratio = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_obj = (torch.min(
                    ratio * adv.unsqueeze(-1),
                    clipped_ratio * adv.unsqueeze(-1)
                ) * mask).sum()
            else:
                pg_obj = (logp * mask * adv.unsqueeze(-1)).sum()
            
            num_valid = mask.sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            
            loss = -pg_obj
            
            # KL penalty
            if ref_logp is not None and args.beta > 0:
                kl = ((logp - ref_logp.detach()) * mask).sum() / num_valid
                loss = loss + args.beta * kl
            
            loss.backward()
        
        rewards_list.append(rewards.mean().item())
        
        # 打印示例（第一轮）- 显示多轮搜索信息
        if step == 0 and master_process:
            sample_text = texts[0]
            num_searches = len(train_task.parse_search_tags(sample_text))
            num_thinks = train_task.count_think_tags(sample_text)
            print0(f"\n Sample completion (reward={rewards[0].item():.3f}, "
                   f"searches={num_searches}, thinks={num_thinks}):")
            print0(f"Question: {item['messages'][-1]['content'][:100]}...")
            print0(f"Expected hops: {item.get('num_hops', '?')}")
            print0(f"Response ({len(sample_text)} chars):")
            # 打印前 500 字符
            for line in sample_text[:500].split("\n"):
                print0(f"  {line}")
            if len(sample_text) > 500:
                print0(f"  ... [{len(sample_text) - 500} more chars]")
    
    # Aggregate rewards across ranks
    mean_reward = sum(rewards_list) / len(rewards_list)
    if ddp:
        mean_reward_t = torch.tensor(mean_reward, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_t, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_t.item()
    
    # Gradient clipping + optimizer step
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
    
    # Logging
    smooth_reward = ema_beta * smooth_reward + (1 - ema_beta) * mean_reward
    debiased_reward = smooth_reward / (1 - ema_beta ** (step + 1))
    
    print0(f"Step {step}/{num_steps} | Epoch {epoch} | reward: {debiased_reward:.4f} | "
           f"lrm: {lrm:.3f} | grad: {grad_norm:.3f} | dt: {dt:.2f}s")
    
    if use_swanlab:
        swanlab.log({
            "step": step,
            "epoch": epoch,
            "train/reward": debiased_reward,
            "train/raw_reward": mean_reward,
            "train/lrm": lrm,
            "train/grad_norm": grad_norm,
            "train/dt": dt,
        })
    
    # Save checkpoint
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        save_dir = os.path.join(args.output_dir, f"step_{step:06d}_epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_hf(save_dir)
        tokenizer.save(save_dir)

        meta = {
            "step": step,
            "epoch": epoch,
            "reward": debiased_reward,
            "config": user_config,
        }
        with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print0(f"[OK] Saved checkpoint to {save_dir}")

        if use_swanlab:
            swanlab.log({"checkpoint/step": step, "checkpoint/reward": debiased_reward})

# Close the generator to avoid cleanup warnings
batch_iterator.close()

# ── 保存最终模型 ──
if master_process:
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_hf(final_dir)
    tokenizer.save(final_dir)
    meta = {
        "step": step,
        "epoch": epoch,
        "reward": debiased_reward,
        "config": user_config,
        "total_training_time_min": total_training_time / 60,
        "peak_memory_mib": get_max_memory() / 1024 / 1024,
    }
    with open(os.path.join(final_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print0(f"[OK] Final model saved to {final_dir}")

# Final stats
print0("=" * 80)
print0(f"Search-R1 GRPO Training complete!")
print0(f"  Final reward: {debiased_reward:.4f}")
print0(f"  Epochs completed: {epoch + 1}")
print0(f"  Total time: {total_training_time / 60:.2f}m")
print0(f"  Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"  Final model: {os.path.join(args.output_dir, 'final')}")
print0("=" * 80)

# Cleanup
if use_swanlab:
    swanlab.finish()
compute_cleanup()


