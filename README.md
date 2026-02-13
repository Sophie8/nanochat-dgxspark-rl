# nanochat-dgxspark-rl

**专为 NVIDIA DGX Spark 打造的轻量级 LLM 强化学习训练框架**

[English](README_EN.md)

一个极简但功能完整的 LLM 训练框架，支持 LoRA SFT 微调、GRPO 强化学习、以及 Search-R1 多轮搜索推理训练。专为单节点和双节点 DGX Spark 集群设计，开箱即用。

---

## 功能特性

| 功能 | 单节点 | 双节点 DDP |
|------|--------|-----------|
| **LoRA SFT 微调** | `sft_single_node.sh` | `sft_2node.sh` |
| **GRPO 强化学习** | `grpo_single_node.sh` | `grpo_2node.sh` |
| **Search-R1 多轮搜索推理** | `search_r1_single_node.sh` | `search_r1_2node.sh` |

- 基于 HuggingFace Transformers + PEFT，支持加载任意 HF 模型（Qwen、LLaMA 等）
- LoRA 参数高效微调（仅训练约 0.5% 的参数）
- 基础模型即参考模型，无需额外显存（通过 `disable_adapter` 切换）
- 多轮搜索 Rollout：模型生成 -> 检测 `<search>` 标签 -> 执行真实搜索 -> 注入结果 -> 继续生成
- 搜索结果 Loss Masking：搜索结果 token 不参与梯度计算
- SwanLab 实验日志记录
- 完整的模型保存与恢复

---

## 项目结构

```
nanochat-dgxspark-rl/
|-- nanochat/                      # nanochat 核心库（来自 nanochat 项目）
|   |-- __init__.py
|   |-- common.py                  # 分布式初始化、print0、DummyWandb
|   |-- hf_model_wrapper.py        # HFModelWrapper + LoRA + DistAdamW
|   |-- hf_tokenizer_wrapper.py    # HFTokenizerWrapper + 特殊 token
|   `-- optim.py                   # MuonAdamW / DistMuonAdamW 优化器
|-- tools/                         # 搜索工具（来自 nanochat/tools）
|   |-- search_tools.py            # 搜索引擎（DuckDuckGo/Tavily/Serper/Gemini/Mock）
|   `-- memory_manager.py          # 对话记忆管理
|-- tasks/                         # 任务定义（来自 nanochat/tasks）
|   |-- common.py                  # 基础 Task 类
|   |-- custom_jsonl.py            # JSONL 格式 RL 任务
|   `-- search_r1.py               # Search-R1 多轮搜索任务
|-- scripts/                       # 训练脚本（与 nanochat 原始脚本保持一致）
|   |-- sft.py                     # LoRA SFT 训练（DDP + 梯度检查点）
|   |-- grpo.py                    # GRPO 强化学习训练（DistAdamW）
|   |-- search_r1_grpo.py          # Search-R1 多轮 GRPO 训练
|   |-- generate_search_r1_data_with_gemini.py   # 用 Gemini 生成训练数据
|   |-- generate_search_r1_multiturn_data.py     # 用 Gemini+DuckDuckGo 生成多轮训练数据
|   `-- gen_val_gemini.py          # 用 Gemini 生成验证数据
|-- examples/                      # DGX Spark 启动脚本（单节点 & 双节点）
|   |-- sft_single_node.sh
|   |-- sft_2node.sh
|   |-- grpo_single_node.sh
|   |-- grpo_2node.sh
|   |-- search_r1_single_node.sh
|   `-- search_r1_2node.sh
|-- data/                          # 示例数据
|   |-- gsm8k_mini_test.jsonl      # GSM8K 数学推理示例
|   |-- search_r1_demo.jsonl       # Search-R1 中文多轮搜索示例
|   `-- sft_demo.jsonl             # SFT 中文对话示例
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

> **与 nanochat 的关系**：本项目从 [nanochat](https://github.com/karpathy/nanochat) 中提取了 HFModelWrapper、HFTokenizerWrapper、compute_init 等核心组件，训练脚本保持与 nanochat 原始代码一致的结构和接口。项目完全自包含，无需额外安装 nanochat 即可运行。

---

## 环境准备

### 方式一：pip 安装

```bash
git clone https://github.com/your-org/nanochat-dgxspark-rl.git
cd nanochat-dgxspark-rl
pip install -e ".[all]"
```

### 方式二：直接安装依赖

```bash
pip install -r requirements.txt
```

### 核心依赖

- Python >= 3.9
- PyTorch >= 2.1.0
- Transformers >= 4.40.0
- PEFT >= 0.11.0
- duckduckgo-search >= 6.0.0（Search-R1 所需）
- swanlab >= 0.3.0（可选，实验日志记录）

---

## 用户配置

在运行训练之前，你需要根据自己的环境修改以下配置：

### 必须配置

| 配置项 | 说明 | 配置方式 |
|--------|------|---------|
| **模型路径** | HuggingFace 模型名称或本地路径 | 脚本参数 `--model-path`，如 `Qwen/Qwen3-8B` 或本地路径 `/path/to/model` |
| **训练数据** | JSONL 格式的训练数据文件 | 脚本参数 `--data`（SFT）、`--task`（GRPO）、`--train-data`（Search-R1） |
| **输出目录** | 模型保存路径 | 脚本参数 `--output-dir`，默认 `outputs/sft` |

### 分布式训练配置（双节点）

| 配置项 | 说明 | 配置方式 |
|--------|------|---------|
| **MASTER_ADDR** | 主节点 IP 地址 | 环境变量，需替换为你的主节点实际 IP |
| **MASTER_PORT** | 通信端口 | 环境变量，默认 `29500` |
| **NODE_RANK** | 节点编号（主节点 0，工作节点 1） | 环境变量 |
| **NCCL_SOCKET_IFNAME** | 网络接口名称 | 环境变量，根据实际网络接口设置（如 `eth0`、`enp226s0`） |

### 搜索引擎配置（Search-R1 需要）

| 配置项 | 说明 | 配置方式 |
|--------|------|---------|
| **搜索代理** | HTTP 代理地址（如需翻墙访问搜索引擎） | 脚本参数 `--search-proxy` 或环境变量 `SEARCH_PROXY` |
| **TAVILY_API_KEY** | Tavily 搜索引擎 API key（可选） | 环境变量 |
| **SERPER_API_KEY** | Serper 搜索引擎 API key（可选） | 环境变量 |
| **GEMINI_API_KEY** | Gemini API key（可选） | 环境变量 |
| **GEMINI_API_URL** | Gemini API 地址（可选） | 环境变量 |

> **搜索引擎优先级**：默认使用 DuckDuckGo（免费，无需 API key）。如需更稳定的搜索服务，可配置 Tavily 或 Serper 的 API key。

### 实验日志配置（可选）

| 配置项 | 说明 | 配置方式 |
|--------|------|---------|
| **SwanLab 模式** | 日志记录模式 | 脚本参数 `--swanlab-mode`（`cloud`/`local`/`offline`/`disabled`） |
| **wandb** | GRPO 脚本支持 wandb | 脚本参数 `--run`（设为 `dummy` 可禁用） |

### 配置示例

```bash
# 最简单的单节点 SFT 训练（只需指定模型和数据）
python scripts/sft.py \
    --model-path /path/to/your/model \
    --data /path/to/your/data.jsonl \
    --output-dir outputs/my_sft

# Search-R1 训练（带搜索代理）
export SEARCH_PROXY=http://your-proxy:port
python scripts/search_r1_grpo.py \
    --model-path /path/to/your/model \
    --train-data /path/to/search_r1_data.jsonl \
    --search-engine duckduckgo \
    --search-proxy $SEARCH_PROXY

# 双节点分布式训练（需在两个节点分别执行）
export MASTER_ADDR=<your-master-node-ip>
export MASTER_PORT=29500
export NODE_RANK=0  # 主节点为 0，工作节点为 1
torchrun --nproc_per_node=1 --nnodes=2 \
    --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/sft.py --model-path /path/to/model --data /path/to/data.jsonl
```

---

## 快速开始

### 一、LoRA SFT 微调

SFT（Supervised Fine-Tuning）使用标注的对话数据对模型进行有监督微调。

**数据格式**（JSONL，每行一条）：

```json
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的？"}]}
```

**单节点训练：**

```bash
python scripts/sft.py \
    --model-path Qwen/Qwen3-8B \
    --data data/sft_demo.jsonl \
    --output-dir outputs/sft \
    --num-epochs 2 \
    --lr 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 1 \
    --grad-accum 4 \
    --max-seq-len 4096

# 或使用启动脚本：
bash examples/sft_single_node.sh
```

**双节点分布式训练（2x DGX Spark）：**

```bash
# 节点 0（主节点）：
MASTER_ADDR=<NODE0_IP> NODE_RANK=0 bash examples/sft_2node.sh

# 节点 1（工作节点）：
MASTER_ADDR=<NODE0_IP> NODE_RANK=1 bash examples/sft_2node.sh
```

训练完成后，LoRA adapter 保存在 `outputs/sft/final/` 目录。

---

### 二、GRPO 强化学习训练

GRPO（Group Relative Policy Optimization）是一种基于组相对优势的策略优化算法，用于通过奖励信号训练模型。

**数据格式**（JSONL）：

```json
{"messages": [{"role": "user", "content": "Natalia卖了48个发夹给朋友..."}], "answer": "72"}
```

**单节点训练：**

```bash
python scripts/grpo.py \
    --model-path Qwen/Qwen3-8B \
    --task data/gsm8k_mini_test.jsonl \
    --output-dir outputs/grpo \
    --num-generations 4 \
    --beta 0.01 \
    --clip-eps 0.2 \
    --lr 5e-5

# 或使用启动脚本：
bash examples/grpo_single_node.sh
```

**双节点分布式训练：**

```bash
# 节点 0：
MASTER_ADDR=<NODE0_IP> NODE_RANK=0 bash examples/grpo_2node.sh

# 节点 1：
MASTER_ADDR=<NODE0_IP> NODE_RANK=1 bash examples/grpo_2node.sh
```

---

### 三、Search-R1 多轮搜索推理训练

Search-R1 是本框架的核心亮点，训练模型自主使用搜索工具进行多步推理：

```
用户: 量子计算和传统计算机的区别是什么？

模型: <think>我需要了解量子计算的基本原理</think>
      <search>量子计算原理 量子比特</search>
      <information>[搜索结果自动注入]</information>
      <think>根据搜索结果，量子计算使用量子比特...还需要对比</think>
      <search>量子计算 vs 传统计算机 优势对比</search>
      <information>[搜索结果自动注入]</information>
      <think>综合以上信息...</think>
      量子计算与传统计算机的主要区别在于...
```

**数据格式**（JSONL）：

```json
{
  "question": "2024年诺贝尔物理学奖授予了谁？",
  "answer": "机器学习, 人工神经网络, John Hopfield, Geoffrey Hinton",
  "num_hops": 2,
  "difficulty": "medium",
  "search_chain": [
    {"query": "2024年诺贝尔物理学奖获得者", "purpose": "查找获奖者"},
    {"query": "Hopfield Hinton 研究贡献", "purpose": "了解具体研究内容"}
  ]
}
```

**单节点训练：**

```bash
python scripts/search_r1_grpo.py \
    --model-path Qwen/Qwen3-8B \
    --train-data data/search_r1_demo.jsonl \
    --search-engine duckduckgo \
    --search-proxy http://<PROXY_IP>:7890 \
    --max-search-turns 5 \
    --num-generations 4 \
    --output-dir outputs/search_r1

# 或使用启动脚本：
bash examples/search_r1_single_node.sh

# 带搜索代理：
SEARCH_PROXY=http://<PROXY_IP>:7890 bash examples/search_r1_single_node.sh
```

**双节点分布式训练：**

```bash
# 节点 0：
MASTER_ADDR=<NODE0_IP> NODE_RANK=0 \
    SEARCH_PROXY=http://<PROXY_IP>:7890 \
    bash examples/search_r1_2node.sh

# 节点 1：
MASTER_ADDR=<NODE0_IP> NODE_RANK=1 \
    SEARCH_PROXY=http://<PROXY_IP>:7890 \
    bash examples/search_r1_2node.sh
```

> **注意**：搜索代理仅用于搜索引擎请求，不会影响 SwanLab 等其他网络服务。

---

## 训练数据生成

本项目提供三个脚本用于生成 Search-R1 训练和验证数据，使用 Gemini API 生成问题，并结合真实搜索引擎获取搜索结果。

### 环境变量配置

所有数据生成脚本需要设置以下环境变量：

```bash
export GEMINI_API_URL="https://your-gemini-api-endpoint/v1/chat/completions"
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-3-flash-preview"  # 可选，默认值已设置
```

### 方式一：Gemini 生成基础训练数据

使用 Gemini 生成问题和答案关键词（不含真实搜索结果）：

```bash
python scripts/generate_search_r1_data_with_gemini.py \
    --output data/search_r1_train_gemini.jsonl \
    --num-examples 200
```

### 方式二：Gemini + DuckDuckGo 生成多轮训练数据（推荐）

三步流程：Gemini 生成问题 -> DuckDuckGo 真实搜索 -> Gemini 合成完整的多轮推理响应：

```bash
python scripts/generate_search_r1_multiturn_data.py \
    --output data/search_r1_multiturn_train.jsonl \
    --num-examples 200 \
    --proxy http://your-proxy:port

# 同时生成验证集：
python scripts/generate_search_r1_multiturn_data.py \
    --output data/search_r1_multiturn_train.jsonl \
    --num-examples 200 \
    --proxy http://your-proxy:port \
    --gen-val \
    --val-output data/search_r1_multiturn_val.jsonl \
    --val-examples 50
```

### 方式三：Gemini 搜索引擎生成验证数据

当 DuckDuckGo 代理不可用时，可以使用 Gemini 模拟搜索引擎生成验证数据：

```bash
python scripts/gen_val_gemini.py
```

> **数据质量说明**：方式二生成的数据质量最高，因为使用了真实搜索引擎的结果。方式一和方式三使用 Gemini 生成/模拟搜索结果，可能存在信息不够准确的情况。

---

## 双节点 DGX Spark 分布式训练详细教程

本项目专为 NVIDIA DGX Spark（GB10 Grace Blackwell）设计，以下是完整的双节点部署流程。

### 硬件架构

```
+-------------------+              +-------------------+
|  DGX Spark #0     |  QSFP/CX7   |  DGX Spark #1     |
|  (主节点)          |<============>|  (工作节点)         |
|  GPU: GB10        |   200Gbps    |  GPU: GB10        |
|  torchrun rank=0  |    NCCL      |  torchrun rank=1  |
+-------------------+              +-------------------+
```

### 步骤 1：硬件连接

按照 NVIDIA 官方教程完成两台 DGX Spark 的物理连接和 NCCL 测试：

1. 堆叠连接：https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks
2. NCCL 测试：https://build.nvidia.com/spark/nccl

### 步骤 2：配置网络环境

**主节点（Node 0）：**

```bash
NODE_RANK=0
IB_IF=$(/usr/sbin/ibdev2netdev | awk '/(Up|ACTIVE)/{print $5; exit}')
MASTER_ADDR=$(ip -o -4 addr show dev "$IB_IF" | awk '{print $4}' | cut -d/ -f1)

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=29500
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=$IB_IF

echo "MASTER_ADDR=$MASTER_ADDR  NODE_RANK=0  IFACE=$IB_IF"
```

**工作节点（Node 1）：**

```bash
NODE_RANK=1
IB_IF=$(/usr/sbin/ibdev2netdev | awk '/(Up|ACTIVE)/{print $5; exit}')
MASTER_ADDR="<主节点的 MASTER_ADDR>"  # 替换为上一步获取的地址

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=29500
export NODE_RANK=1
export NCCL_SOCKET_IFNAME=$IB_IF
```

### 步骤 3：启动训练

在两个节点上分别执行（以 SFT 为例）：

```bash
# 两个节点都执行：
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=29500 \
    scripts/sft.py \
    --model-path Qwen/Qwen3-8B \
    --data data/sft_demo.jsonl \
    --output-dir outputs/sft_2node
```

> 训练会在两个节点都启动后才开始。工作节点可能会短暂显示连接警告，这是正常的。

### NCCL 调优参数

```bash
export NCCL_IB_DISABLE=0          # 启用 InfiniBand
export NCCL_SOCKET_IFNAME=eth0    # 网络接口
export NCCL_DEBUG=INFO            # 调试日志
export NCCL_P2P_DISABLE=0         # 启用 P2P
```

---

## 关键参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | 必填 | HuggingFace 模型名称或本地路径 |
| `--lora-r` | 16 | LoRA 秩 |
| `--lora-alpha` | 32 | LoRA alpha |
| `--lr` | 5e-5 | 学习率 |
| `--dtype` | bfloat16 | 训练精度 |
| `--swanlab-mode` | cloud | SwanLab 日志模式（cloud/local/offline/disabled） |

### GRPO 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-generations` | 4 | 每个 prompt 生成的 completion 数量（G） |
| `--beta` | 0.01 | KL 散度惩罚系数 |
| `--clip-eps` | 0.2 | PPO 风格裁剪范围 |
| `--advantage-norm` | zscore | 优势归一化方式 |

### Search-R1 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--search-engine` | duckduckgo | 搜索引擎后端 |
| `--search-proxy` | 无 | 搜索代理（仅搜索引擎使用） |
| `--max-search-turns` | 5 | 每次 completion 最大搜索轮数 |

---

## 算法说明

### GRPO 算法流程

每个优化步骤：

1. 从数据集采样 prompt
2. 对每个 prompt，使用 LoRA 策略模型生成 G 个 completion
3. 使用奖励函数对每个 completion 评分
4. 计算组相对优势（z-score 归一化）
5. 计算策略模型的 per-token log prob
6. 禁用 LoRA adapter 计算参考模型的 log prob（无需额外显存）
7. PPO 风格裁剪的策略梯度 + KL 惩罚
8. 仅更新 LoRA 参数

### Search-R1 多轮 Rollout 流程

```
prompt --> generate_segment()
  |-- 未检测到 <search> 标签 --> 结束（最终答案）
  `-- 检测到 <search>查询</search>
        |-- execute_search(查询) --> 调用真实搜索引擎
        |-- 注入 <information>搜索结果</information>
        |-- 标记 search_mask[info_tokens] = 0（从 loss 中排除）
        `-- 继续 generate_segment()
              `-- 重复直到达到 MAX_SEARCH_TURNS
```

---

## 致谢

本项目的诞生离不开以下项目和社区的支持：

- [nanochat](https://github.com/nanochat/nanochat) -- 本项目基于 nanochat 的核心架构构建，感谢 nanochat 提供的优秀基础设施
- [Train nanochat on 2 NVIDIA DGX Sparks](https://gist.github.com/emaadmanzoor/d245c0c0ce90b25b4d50c0ffc448f876) -- 感谢 @emaadmanzoor 提供的双节点 DGX Spark 分布式训练教程，本项目的分布式训练方案参考了该教程
- [NVIDIA DGX Spark 社区](https://build.nvidia.com/spark) -- 感谢 NVIDIA DGX Spark 社区的技术支持和硬件文档
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) -- 搜索增强推理的原始框架
- [TRL](https://github.com/huggingface/trl) -- GRPO 算法参考实现
- [PEFT](https://github.com/huggingface/peft) -- LoRA 实现
- [SwanLab](https://swanlab.cn) -- 实验日志记录平台

---

## 项目由来

本项目的开发流程如下：首先在本地对 nanochat 项目进行修改和扩展，增加了 HuggingFace 模型 LoRA 微调、GRPO 强化学习、Search-R1 多轮搜索推理等功能，并在 NVIDIA DGX Spark 上完成了单节点和双节点分布式训练的验证。在确认所有功能运行成功之后，借助 AI 工具将这些经过验证的代码整理、重构为当前的独立开源项目。

---

## 许可证

Apache 2.0
