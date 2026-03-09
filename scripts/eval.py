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
import torch
from peft import PeftModel

from nanochat.hf_tokenizer_wrapper import HFTokenizerWrapper
from nanochat.hf_model_wrapper import HFModelWrapper


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate a trained model")
	parser.add_argument("--model-path", required=True,
						help="HF model name or local directory containing the model")
	parser.add_argument("--device", default=None,
						help="Device to run on (default: cuda if available, else cpu)")
	# no longer accept a one-off query; this tool is purely interactive
	# (Ctrl+C to exit)
	parser.add_argument("--max-new-tokens", type=int, default=128,
						help="Maximum number of tokens to generate")
	parser.add_argument("--temperature", type=float, default=1.0,
						help="Sampling temperature")
	parser.add_argument("--top-k", type=int, default=50,
						help="Top-k sampling filter (0 = disabled)")
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

	def generate_text(prompt: str) -> str:
		# always prefix with BOS token
		bos_id = tokenizer.get_bos_token_id()
		input_ids = tokenizer.encode(prompt, prepend=bos_id)
		input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

		with torch.no_grad():
			output = model.generate(
				input_tensor,
				max_new_tokens=args.max_new_tokens,
				temperature=args.temperature,
				top_k=args.top_k if args.top_k > 0 else None,
				do_sample=True,
				pad_token_id=tokenizer.hf_tokenizer.pad_token_id,
				eos_token_id=tokenizer.hf_tokenizer.eos_token_id,
			)

		gen_ids = output[0, len(input_ids) :].tolist()
		eos_id = tokenizer.hf_tokenizer.eos_token_id
		if eos_id is not None and eos_id in gen_ids:
			gen_ids = gen_ids[: gen_ids.index(eos_id)]
		return tokenizer.decode(gen_ids)

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

