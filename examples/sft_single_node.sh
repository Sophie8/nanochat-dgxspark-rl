#!/bin/bash
# =============================================================================
# nanochat-dgxspark-rl: Single-Node LoRA SFT Training
# =============================================================================
# Usage:
#   bash examples/sft_single_node.sh
#
# Prerequisites:
#   pip install -r requirements.txt
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATA="${DATA:-data/sft_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_single}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

echo "============================================"
echo " nanochat-dgxspark-rl — Single-Node SFT"
echo "============================================"
echo " Model:     ${MODEL_PATH}"
echo " Data:      ${DATA}"
echo " Output:    ${OUTPUT_DIR}"
echo "============================================"

python scripts/sft.py \
    --model-path "${MODEL_PATH}" \
    --data "${DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --max-seq-len ${MAX_SEQ_LEN} \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --swanlab-mode cloud \
    --swanlab-project nanochat-rl-sft

echo "SFT training complete! Model saved to ${OUTPUT_DIR}"

