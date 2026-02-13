#!/bin/bash
# =============================================================================
# nanochat-dgxspark-rl: 2-Node Distributed GRPO RL Training (DGX Spark)
# =============================================================================
# Usage:
#   # On Node 0 (master):
#   MASTER_ADDR=<NODE0_IP> NODE_RANK=0 bash examples/grpo_2node.sh
#
#   # On Node 1 (worker):
#   MASTER_ADDR=<NODE0_IP> NODE_RANK=1 bash examples/grpo_2node.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

: "${MASTER_ADDR:?Set MASTER_ADDR to the IP of Node 0}"
: "${NODE_RANK:?Set NODE_RANK to 0 (master) or 1 (worker)}"

MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES=2

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TASK="${TASK:-data/gsm8k_mini_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_2node}"
EXAMPLES_PER_STEP="${EXAMPLES_PER_STEP:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"

echo "============================================"
echo " nanochat-dgxspark-rl — 2-Node Distributed GRPO RL"
echo "============================================"
echo " Master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo " Node rank: ${NODE_RANK}/${NNODES}"
echo " Model:     ${MODEL_PATH}"
echo "============================================"

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/grpo.py \
    --model-path "${MODEL_PATH}" \
    --task "${TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --examples-per-step ${EXAMPLES_PER_STEP} \
    --num-generations ${NUM_GENERATIONS} \
    --swanlab-mode cloud \
    --swanlab-project nanochat-rl-grpo

echo "Distributed GRPO training complete on Node ${NODE_RANK}!"

