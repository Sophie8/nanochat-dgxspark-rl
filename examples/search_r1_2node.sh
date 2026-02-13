#!/bin/bash
# =============================================================================
# nanochat-dgxspark-rl: Search-R1 2-Node Distributed GRPO (DGX Spark)
# =============================================================================
# Usage:
#   # On Node 0 (master):
#   MASTER_ADDR=<NODE0_IP> NODE_RANK=0 bash examples/search_r1_2node.sh
#
#   # On Node 1 (worker):
#   MASTER_ADDR=<NODE0_IP> NODE_RANK=1 bash examples/search_r1_2node.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

: "${MASTER_ADDR:?Set MASTER_ADDR to the IP of Node 0}"
: "${NODE_RANK:?Set NODE_RANK to 0 (master) or 1 (worker)}"

MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES=2

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-data/search_r1_train.jsonl}"
VAL_DATA="${VAL_DATA:-data/search_r1_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/search_r1_2node}"
SEARCH_ENGINE="${SEARCH_ENGINE:-duckduckgo}"
SEARCH_PROXY="${SEARCH_PROXY:-}"

echo "============================================"
echo " nanochat-dgxspark-rl — Search-R1 2-Node GRPO"
echo "============================================"
echo " Master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo " Node rank: ${NODE_RANK}/${NNODES}"
echo " Model:     ${MODEL_PATH}"
echo " Engine:    ${SEARCH_ENGINE}"
echo "============================================"

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

PROXY_ARGS=""
if [ -n "${SEARCH_PROXY}" ]; then
    PROXY_ARGS="--search-proxy ${SEARCH_PROXY}"
fi

VAL_ARGS=""
if [ -f "${VAL_DATA}" ]; then
    VAL_ARGS="--val-data ${VAL_DATA}"
fi

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/search_r1_grpo.py \
    --model-path "${MODEL_PATH}" \
    --train-data "${TRAIN_DATA}" \
    ${VAL_ARGS} \
    --output-dir "${OUTPUT_DIR}" \
    --search-engine "${SEARCH_ENGINE}" \
    ${PROXY_ARGS} \
    --examples-per-step 4 \
    --num-generations 4 \
    --max-search-turns 5 \
    --swanlab-mode cloud \
    --swanlab-project nanochat-rl-search-r1

echo "Distributed Search-R1 training complete on Node ${NODE_RANK}!"

