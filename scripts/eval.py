"""Simple evaluation script for a trained HuggingFace-compatible model.

This script is intentionally lightweight and only depends on the
`nanochat` wrappers so that it can be used outside of the full training
pipeline.  It loads a model+tokenizer, accepts a single query (or drops into
interactive REPL if no query is provided), and prints the generated
completion.

Example (one-shot):

	python scripts/eval.py \
		--model-path /path/to/Qwen3-8B \
		--query "What is the capital of France?" \
		--max-new-tokens 64

Interactive mode:

	python scripts/eval.py --model-path /path/to/Qwen3-8B
	>>> Hello, how are you?
	I'm fine, thank you!
	>>> exit
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from peft import PeftModel
from contextlib import nullcontext

# Add parent directory to path for task imports
sys.path.append(str(Path(__file__).parent.parent))

from nanochat.hf_tokenizer_wrapper import HFTokenizerWrapper
from nanochat.hf_model_wrapper import HFModelWrapper
from tasks.search_r1 import SearchR1Task


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate a trained model with multi-turn search")
	parser.add_argument("--model-path", required=True,
						help="HF model name or local directory containing the model")
	parser.add_argument("--device", default=None,
						help="Device to run on (default: cuda if available, else cpu)")
	parser.add_argument("--max-new-tokens", type=int, default=512,
						help="Maximum number of tokens to generate")
	parser.add_argument("--temperature", type=float, default=0.7,
						help="Sampling temperature")
	parser.add_argument("--top-k", type=int, default=50,
						help="Top-k sampling filter (0 = disabled)")
	parser.add_argument("--max-search-turns", type=int, default=5,
						help="Maximum search-inject turns per completion")
	parser.add_argument("--search-engine", type=str, default="duckduckgo",
						choices=["gemini", "tavily", "serper", "duckduckgo", "mock"],
						help="Search engine for tool calls")
	parser.add_argument("--search-proxy", type=str, default=None,
						help="HTTP proxy for search engine")
	return parser.parse_args()


def main():
	args = parse_args()

	# choose device
	if args.device:
		device = torch.device(args.device)
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# load tokenizer and model wrappers
	tokenizer = HFTokenizerWrapper(args.model_path)
	model = HFModelWrapper(args.model_path, device=device)
	
	# check if LoRA adapter exists and load it
	adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
	if os.path.exists(adapter_config_path):
		print(f"Detected LoRA adapter, loading from {args.model_path}...")
		model.hf_model = PeftModel.from_pretrained(
			model.hf_model,
			args.model_path,
			device_map=str(device),
		)
		# merge adapter into base model for inference
		model.hf_model = model.hf_model.merge_and_unload()
		print("LoRA adapter merged and ready for inference.")
	
	# ensure tokenizer and model vocab sizes match
	model.sync_vocab_size(tokenizer.get_vocab_size())

	# Initialize search task for multi-turn generation
	train_task = SearchR1Task(
		data_path=None,
		search_engine=args.search_engine,
		search_proxy=args.search_proxy,
	)

	@torch.no_grad()
	def generate_multiturn_text(prompt: str) -> str:
		"""Generate text with multi-turn search capability (like Search-R1)."""
		# Prepare prompt tokens
		bos_id = tokenizer.get_bos_token_id()
		prompt_ids = tokenizer.encode(prompt, prepend=bos_id)
		
		# Constants from search_r1_grpo.py
		MAX_SEARCH_TURNS = args.max_search_turns
		SEARCH_TAG_START = "<search>"
		SEARCH_TAG_END = "</search>"
		INFO_TAG_START = "<information>"
		INFO_TAG_END = "</information>"
		
		context_ids = list(prompt_ids)
		all_new_ids = []
		search_result_ranges = []
		tokens_remaining = args.max_new_tokens
		
		for turn in range(MAX_SEARCH_TURNS):
			if tokens_remaining <= 0:
				break
			
			# Generate a segment
			max_seg_tokens = min(tokens_remaining, args.max_new_tokens // 2)
			input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
			
			with torch.no_grad():
				output_ids = model.generate(
					input_tensor,
					max_new_tokens=max_seg_tokens,
					temperature=args.temperature,
					top_k=args.top_k if args.top_k > 0 else None,
					do_sample=args.temperature > 0,
					pad_token_id=tokenizer.hf_tokenizer.pad_token_id,
					eos_token_id=tokenizer.hf_tokenizer.eos_token_id,
				)
			
			seg_ids = output_ids[0, len(context_ids):].tolist()
			eos_id = tokenizer.hf_tokenizer.eos_token_id
			if eos_id is not None and eos_id in seg_ids:
				seg_ids = seg_ids[:seg_ids.index(eos_id) + 1]
			
			seg_text = tokenizer.decode(seg_ids)
			
			# Check for search queries
			if SEARCH_TAG_START in seg_text and SEARCH_TAG_END in seg_text:
				# Extract search queries
				start = seg_text.find(SEARCH_TAG_START) + len(SEARCH_TAG_START)
				end = seg_text.find(SEARCH_TAG_END)
				queries_text = seg_text[start:end].strip()
				queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
				
				# Add segment (excluding closing tag for now)
				close_pos = seg_text.find(SEARCH_TAG_END)
				if close_pos >= 0:
					seg_text_before_close = seg_text[:close_pos]
					all_new_ids.extend(tokenizer.encode(seg_text_before_close))
				
				# Execute search and inject results
				for query in queries:
					search_result = train_task.execute_search(query)
					info_text = f"\n{INFO_TAG_START}{search_result}{INFO_TAG_END}\n"
					info_ids = tokenizer.encode(info_text)
					
					# Track search result range
					start_pos = len(all_new_ids)
					all_new_ids.extend(info_ids)
					search_result_ranges.append((start_pos, start_pos + len(info_ids)))
					
					tokens_remaining -= len(info_ids)
				
				# Update context for next generation
				context_ids = list(prompt_ids) + all_new_ids
				tokens_remaining -= len(seg_ids)
			else:
				# No search tag -> final segment
				all_new_ids.extend(seg_ids)
				tokens_remaining -= len(seg_ids)
				break
		
		# Build search mask for reference only (not used in eval)
		search_mask = [1] * len(all_new_ids)
		for start, end in search_result_ranges:
			for i in range(start, min(end, len(search_mask))):
				search_mask[i] = 0
		
		full_text = tokenizer.decode(all_new_ids)
		return full_text

	def generate_text(prompt: str) -> str:
		"""Wrapper to generate text using multi-turn capability."""
		return generate_multiturn_text(prompt)

	# always interactive: load the model once then loop until ctrl+c
	print(f"Loaded model from {args.model_path} on {device}")
	print("Enter prompt and press Enter; press Ctrl+C to exit.")
	try:
		while True:
			try:
				q = input(">>> ")
			except EOFError:
				break
			if not q.strip():
				continue
			print(generate_text(q))
	except KeyboardInterrupt:
		print("\nExiting.")


if __name__ == "__main__":
	main()

