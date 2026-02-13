#!/bin/bash
# =============================================================================
# nanochat-dgxspark-rl: Search-R1 Multi-Turn GRPO Training (Single Node)
# =============================================================================
# Trains the model to use search tools through multi-turn RL.
# The model learns: think -> search -> read results -> think -> answer
#
# Usage:
#   bash examples/search_r1_single_node.sh
#
# With proxy for DuckDuckGo:
#   SEARCH_PROXY=http://<PROXY_IP>:7890 bash examples/search_r1_single_node.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-data/search_r1_train.jsonl}"
VAL_DATA="${VAL_DATA:-data/search_r1_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/search_r1}"
SEARCH_ENGINE="${SEARCH_ENGINE:-duckduckgo}"
SEARCH_PROXY="${SEARCH_PROXY:-}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
EXAMPLES_PER_STEP="${EXAMPLES_PER_STEP:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_SEARCH_TURNS="${MAX_SEARCH_TURNS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"

echo "============================================"
echo " nanochat-dgxspark-rl — Search-R1 GRPO"
echo "============================================"
echo " Model:         ${MODEL_PATH}"
echo " Train data:    ${TRAIN_DATA}"
echo " Search engine: ${SEARCH_ENGINE}"
echo " Search proxy:  ${SEARCH_PROXY:-none}"
echo " Max turns:     ${MAX_SEARCH_TURNS}"
echo "============================================"

PROXY_ARGS=""
if [ -n "${SEARCH_PROXY}" ]; then
    PROXY_ARGS="--search-proxy ${SEARCH_PROXY}"
fi

VAL_ARGS=""
if [ -f "${VAL_DATA}" ]; then
    VAL_ARGS="--val-data ${VAL_DATA}"
fi

python scripts/search_r1_grpo.py \
    --model-path "${MODEL_PATH}" \
    --train-data "${TRAIN_DATA}" \
    ${VAL_ARGS} \
    --output-dir "${OUTPUT_DIR}" \
    --search-engine "${SEARCH_ENGINE}" \
    ${PROXY_ARGS} \
    --num-epochs ${NUM_EPOCHS} \
    --examples-per-step ${EXAMPLES_PER_STEP} \
    --num-generations ${NUM_GENERATIONS} \
    --max-search-turns ${MAX_SEARCH_TURNS} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --swanlab-mode cloud \
    --swanlab-project nanochat-rl-search-r1

echo "Search-R1 training complete! Model saved to ${OUTPUT_DIR}"

