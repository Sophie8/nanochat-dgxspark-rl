#!/bin/bash
# =============================================================================
# nanochat-dgxspark-rl: Single-Node GRPO RL Training
# =============================================================================
# Usage:
#   bash examples/grpo_single_node.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TASK="${TASK:-data/gsm8k_mini_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_single}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
EXAMPLES_PER_STEP="${EXAMPLES_PER_STEP:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LR="${LR:-5e-5}"

echo "============================================"
echo " nanochat-dgxspark-rl — Single-Node GRPO RL"
echo "============================================"
echo " Model:     ${MODEL_PATH}"
echo " Task:      ${TASK}"
echo " Output:    ${OUTPUT_DIR}"
echo "============================================"

python scripts/grpo.py \
    --model-path "${MODEL_PATH}" \
    --task "${TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs ${NUM_EPOCHS} \
    --examples-per-step ${EXAMPLES_PER_STEP} \
    --num-generations ${NUM_GENERATIONS} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --lr ${LR} \
    --swanlab-mode cloud \
    --swanlab-project nanochat-rl-grpo

echo "GRPO training complete! Model saved to ${OUTPUT_DIR}"

