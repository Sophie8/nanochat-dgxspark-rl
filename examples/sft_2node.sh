#!/bin/bash
# =============================================================================
# nanochat-dgxspark-rl: 2-Node Distributed LoRA SFT Training (DGX Spark)
# =============================================================================
# Designed for 2x NVIDIA DGX Spark (GB10 Grace Blackwell, 1 GPU each)
# Connected via high-speed network (200Gbps+)
#
# Usage:
#   # On Node 0 (master):
#   MASTER_ADDR=<NODE0_IP> NODE_RANK=0 bash examples/sft_2node.sh
#
#   # On Node 1 (worker):
#   MASTER_ADDR=<NODE0_IP> NODE_RANK=1 bash examples/sft_2node.sh
#
# Environment Variables:
#   MASTER_ADDR  - IP address of Node 0 (required)
#   NODE_RANK    - 0 for master, 1 for worker (required)
#   MASTER_PORT  - Communication port (default: 29500)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# Required env vars
: "${MASTER_ADDR:?Set MASTER_ADDR to the IP of Node 0}"
: "${NODE_RANK:?Set NODE_RANK to 0 (master) or 1 (worker)}"

MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES=2

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATA="${DATA:-data/sft_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_2node}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

echo "============================================"
echo " nanochat-dgxspark-rl — 2-Node Distributed SFT"
echo "============================================"
echo " Master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo " Node rank: ${NODE_RANK}/${NNODES}"
echo " GPUs/node: ${NPROC_PER_NODE}"
echo " Model:     ${MODEL_PATH}"
echo " Data:      ${DATA}"
echo "============================================"

# NCCL tuning for DGX Spark
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/sft.py \
    --model-path "${MODEL_PATH}" \
    --data "${DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --swanlab-mode cloud \
    --swanlab-project nanochat-rl-sft

echo "Distributed SFT training complete on Node ${NODE_RANK}!"

