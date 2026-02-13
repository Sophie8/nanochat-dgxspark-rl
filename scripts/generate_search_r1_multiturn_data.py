#!/usr/bin/env python3
"""
生成 Search-R1 多轮交错式（think/search/information）中文训练数据
使用真实搜索引擎获取 <information> 内容

流程:
  1. Gemini 生成问题 + 搜索查询链
  2. DuckDuckGo 真实搜索获取每个查询的结果
  3. Gemini 基于真实搜索结果合成 demo_response 和 answer

Usage:
    python -m scripts.generate_search_r1_multiturn_data \
        --output data/search_r1_multiturn_train.jsonl \
        --num-examples 200 \
        --proxy http://your-proxy:port
"""

import argparse
import json
import os
import sys
import time
import random
from typing import List, Dict, Optional
import requests

# 确保输出实时刷新
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ── API 配置 (通过环境变量设置) ──
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def call_gemini(messages: List[Dict], temperature: float = 0.8,
                max_tokens: int = 2000, retries: int = 3) -> Optional[str]:
    """调用 Gemini API（带重试）"""
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(GEMINI_API_URL, headers=headers,
                                 json=data, timeout=45)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API 调用失败 (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── 真实搜索 ──
import signal
import threading
import concurrent.futures


def real_search(query: str, num_results: int = 3, proxy: Optional[str] = None,
                timeout: int = 15) -> List[Dict]:
    """用 DuckDuckGo 执行真实搜索（带超时保护）"""

    def _do_search():
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        kwargs = {}
        if proxy:
            kwargs["proxy"] = proxy

        ddgs = DDGS(**kwargs)
        results_raw = list(ddgs.text(query, max_results=num_results))
        results = []
        for item in results_raw:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", item.get("url", "")),
                "snippet": item.get("body", item.get("snippet", "")),
            })
        return results

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_search)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"  搜索超时 ({timeout}s) [{query[:40]}]")
        return []
    except Exception as e:
        print(f"  搜索失败 [{query[:40]}]: {e}")
        return []


def format_search_results(results: List[Dict]) -> str:
    """格式化搜索结果为文本块"""
    if not results:
        return "未找到相关搜索结果。"
    parts = []
    for i, r in enumerate(results, 1):
        snippet = r["snippet"][:200]
        parts.append(f"[{i}] {r['title']}\n{snippet}")
    return "\n".join(parts)


# ── 多跳问题类别 ──
MULTIHOP_CATEGORIES = [
    {
        "category": "人物关联",
        "difficulty": "medium",
        "num_hops": 2,
        "description": "需要先搜索某个人物/事件，再根据第一次结果搜索关联信息",
        "examples": [
            "2024年诺贝尔物理学奖得主之一杰弗里·辛顿，他在哪所大学任教？该大学的现任校长是谁？",
            "《三体》的作者刘慈欣是哪里人？他的家乡有什么著名的旅游景点？",
        ],
    },
    {
        "category": "因果链推理",
        "difficulty": "hard",
        "num_hops": 3,
        "description": "需要逐步搜索因果链条，A导致B，B又导致C",
        "examples": [
            "2024年美联储降息对中国A股市场产生了什么影响？这种影响又如何改变了普通投资者的理财策略？",
            "全球芯片短缺最初由什么事件引发？这场短缺如何影响了中国汽车产业？中国采取了什么应对措施？",
        ],
    },
    {
        "category": "比较多源",
        "difficulty": "hard",
        "num_hops": 2,
        "description": "需要分别搜索两个实体的信息，然后进行对比分析",
        "examples": [
            "比较华为Mate 70和苹果iPhone 16的主要性能参数，哪个更适合摄影爱好者？",
            "中国高铁和日本新干线在最高时速、覆盖里程和票价方面有何不同？",
        ],
    },
    {
        "category": "时间线追踪",
        "difficulty": "medium",
        "num_hops": 2,
        "description": "需要搜索某事件的发展时间线，追踪从起因到最新进展",
        "examples": [
            "中国空间站天宫号从立项到建成经历了哪些关键节点？最新的载人飞行任务是什么？",
            "OpenAI从成立到发布GPT-4o经历了哪些重要里程碑？最新的产品动态是什么？",
        ],
    },
    {
        "category": "概念与实例",
        "difficulty": "easy",
        "num_hops": 2,
        "description": "先搜索一个概念/技术的解释，再搜索具体应用实例",
        "examples": [
            "什么是大语言模型的RAG技术？请举一个实际应用RAG的中国公司的例子。",
            "什么是碳中和？中国在碳中和方面有哪些具体的政策和成就？",
        ],
    },
    {
        "category": "地理文化关联",
        "difficulty": "easy",
        "num_hops": 2,
        "description": "先搜索某个地理/文化实体，再搜索其关联的深层信息",
        "examples": [
            "世界上最深的湖泊在哪个国家？该国的首都是哪里？有什么著名的文化传统？",
            "丝绸之路的起点和终点分别是哪里？现代'一带一路'倡议涵盖了多少个国家？",
        ],
    },
    {
        "category": "科技前沿追踪",
        "difficulty": "hard",
        "num_hops": 3,
        "description": "追踪某项前沿科技的原理、最新进展和未来影响",
        "examples": [
            "量子纠错技术的基本原理是什么？谷歌Willow芯片在量子纠错方面取得了什么突破？这对实用化量子计算意味着什么？",
            "脑机接口技术的核心原理是什么？Neuralink最新的人体试验进展如何？中国在这一领域有哪些研究成果？",
        ],
    },
    {
        "category": "政策影响分析",
        "difficulty": "medium",
        "num_hops": 2,
        "description": "先搜索某个政策的内容，再搜索该政策的实际影响和评价",
        "examples": [
            "中国最新的新能源汽车补贴政策具体内容是什么？该政策对国产新能源品牌销量产生了怎样的影响？",
            "欧盟的《人工智能法案》主要规定了什么？这对中国AI企业出海有何影响？",
        ],
    },
]


# ── 第一步：生成问题和搜索链 ──
def generate_questions_batch(category_info: Dict, batch_size: int = 5) -> List[Dict]:
    """用 Gemini 生成问题+搜索查询链（不生成 demo_response）"""

    category = category_info["category"]
    difficulty = category_info["difficulty"]
    num_hops = category_info["num_hops"]
    description = category_info["description"]
    examples_str = "\n".join(f"  - {e}" for e in category_info["examples"])

    prompt = f"""你是一个专业的训练数据生成器。请生成 {batch_size} 条中文多跳搜索问题。

## 要求
类别：{category}
难度：{difficulty}
需要的搜索跳数：{num_hops}
说明：{description}

参考示例：
{examples_str}

## 输出格式
对于每条数据，请生成：
1. **question**: 一个需要 {num_hops} 次搜索才能完整回答的中文问题
2. **search_chain**: 一个数组，包含 {num_hops} 个搜索步骤，每步有：
   - query: 适合搜索引擎的搜索查询（简洁、精确）
   - purpose: 为什么要搜索这个

## 注意事项
- 问题必须需要多步搜索才能回答
- 搜索查询要简洁明确，适合在搜索引擎中使用
- 问题要多样化，不要重复
- 确保 JSON 格式正确

请严格按以下JSON格式返回（不要添加任何其他文字）：
{{"data": [
  {{
    "question": "...",
    "search_chain": [
      {{"query": "搜索查询1", "purpose": "目的1"}},
      {{"query": "搜索查询2", "purpose": "目的2"}}
    ],
    "num_hops": {num_hops},
    "difficulty": "{difficulty}"
  }},
  ...
]}}"""

    messages = [
        {"role": "system", "content": (
            "你是一个AI训练数据生成器。生成需要多轮搜索才能回答的中文问题。"
            "只返回有效的JSON格式，不要添加其他文字。"
        )},
        {"role": "user", "content": prompt},
    ]

    response = call_gemini(messages, temperature=0.9, max_tokens=3000)
    if not response:
        return []

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        items = parsed.get("data", [])

        valid = []
        for item in items:
            if not all(k in item for k in ("question", "search_chain")):
                continue
            if not isinstance(item["search_chain"], list) or len(item["search_chain"]) < 1:
                continue
            valid.append({
                "question": item["question"],
                "search_chain": item["search_chain"],
                "num_hops": item.get("num_hops", num_hops),
                "difficulty": difficulty,
            })
        return valid

    except Exception as e:
        print(f"  解析失败: {e}")
        return []


# ── 第二步：执行真实搜索 ──
def execute_search_chain(search_chain: List[Dict], proxy: Optional[str] = None) -> List[Dict]:
    """对搜索链中的每个查询执行真实搜索"""
    search_results = []
    for step in search_chain:
        query = step["query"]
        results = real_search(query, num_results=3, proxy=proxy)
        formatted = format_search_results(results)
        search_results.append({
            "query": query,
            "purpose": step.get("purpose", ""),
            "results": results,
            "formatted": formatted,
        })
        time.sleep(0.5)  # 避免搜索速率限制
    return search_results


# ── 第三步：基于真实搜索结果合成完整数据 ──
def synthesize_response(question: str, search_chain_with_results: List[Dict],
                        num_hops: int) -> Optional[Dict]:
    """基于真实搜索结果，让 Gemini 合成 demo_response 和 answer"""

    # 构建搜索结果描述
    search_info = ""
    for i, step in enumerate(search_chain_with_results, 1):
        search_info += f"\n--- 第 {i} 次搜索 ---\n"
        search_info += f"查询: {step['query']}\n"
        search_info += f"目的: {step['purpose']}\n"
        search_info += f"真实搜索结果:\n{step['formatted']}\n"

    prompt = f"""基于以下问题和真实搜索引擎返回的结果，请生成一个完整的多轮推理响应。

## 问题
{question}

## 真实搜索结果
{search_info}

## 要求
1. 生成 **answer**: 最终答案的关键词（2-5个，逗号分隔），必须基于真实搜索结果
2. 生成 **demo_response**: 使用以下标签格式展示多轮推理过程：

```
<think>首先，我需要了解...让我搜索一下。</think>
<search>第一次搜索查询</search>
<information>基于真实搜索结果的摘要（2-4句话，必须引用真实搜索结果中的事实和数据）</information>
<think>根据搜索结果，我了解到...但还需要进一步了解...让我继续搜索。</think>
<search>第二次搜索查询</search>
<information>基于真实搜索结果的摘要（2-4句话）</information>
<think>综合以上信息，我可以得出结论...</think>
最终答案内容（基于真实搜索结果总结）
```

## 关键要求
- <information> 中的内容必须**忠实于**提供的真实搜索结果，不要编造不存在的数据
- <search> 标签中的查询应该使用搜索链中的实际查询
- <think> 部分包含反思性短语
- 如果搜索结果为空或不相关，在 think 中说明并适当调整

请严格按以下JSON格式返回：
{{"answer": "关键词1, 关键词2, ...", "demo_response": "<think>...</think>\\n<search>...</search>\\n..."}}"""

    messages = [
        {"role": "system", "content": (
            "你是一个训练数据合成器。根据真实搜索结果生成多轮推理响应。"
            "只返回有效的JSON格式。<information>标签的内容必须忠实于提供的真实搜索结果。"
        )},
        {"role": "user", "content": prompt},
    ]

    response = call_gemini(messages, temperature=0.5, max_tokens=3000)
    if not response:
        return None

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        answer = parsed.get("answer", "")
        demo = parsed.get("demo_response", "")

        if not answer or not demo:
            return None
        if "<think>" not in demo or "<search>" not in demo:
            return None

        return {"answer": answer, "demo_response": demo}

    except Exception as e:
        print(f"  合成失败: {e}")
        return None


# ── 主流程 ──
def generate_dataset(output_path: str, num_examples: int = 200,
                     proxy: Optional[str] = None):
    """生成完整的多轮中文训练数据集（使用真实搜索）"""

    print("=" * 80)
    print("Search-R1 多轮中文训练数据生成 (真实搜索引擎版)")
    print(f"  目标数量: {num_examples}")
    print(f"  输出路径: {output_path}")
    print(f"  搜索代理: {proxy or '无'}")
    print(f"  类别数量: {len(MULTIHOP_CATEGORIES)}")
    print(f"  模型: {GEMINI_MODEL}")
    print("=" * 80)

    # 先测试搜索引擎
    print("\n测试搜索引擎...")
    test_results = real_search("Python programming language", num_results=2, proxy=proxy)
    if test_results:
        print(f"[OK] 搜索引擎正常，返回 {len(test_results)} 条结果")
    else:
        print("[FAIL] 搜索引擎不可用，请检查代理设置！")
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_items: List[Dict] = []

    num_cats = len(MULTIHOP_CATEGORIES)
    per_cat = num_examples // num_cats
    remainder = num_examples % num_cats

    for idx, cat_info in enumerate(MULTIHOP_CATEGORIES):
        target = per_cat + (1 if idx < remainder else 0)
        cat_name = cat_info["category"]
        difficulty = cat_info["difficulty"]
        hops = cat_info["num_hops"]

        print(f"\n[{idx+1}/{num_cats}] {cat_name} (难度:{difficulty}, "
              f"跳数:{hops}, 目标:{target})")

        collected: List[Dict] = []
        attempts = 0
        max_attempts = 15

        while len(collected) < target and attempts < max_attempts:
            need = min(5, target - len(collected))
            print(f"  第 {attempts+1} 轮, 已有 {len(collected)}/{target}, "
                  f"请求 {need} 条...")

            # 第一步：生成问题
            questions = generate_questions_batch(cat_info, batch_size=need)
            if not questions:
                print(f"  [FAIL] 问题生成失败")
                attempts += 1
                time.sleep(1)
                continue

            print(f"  [OK] 生成 {len(questions)} 个问题, 开始真实搜索...")

            # 第二步 + 第三步：搜索 + 合成
            for q_item in questions:
                if len(collected) >= target:
                    break

                question = q_item["question"]
                print(f"    搜索: {question[:50]}...")

                # 执行真实搜索
                search_results = execute_search_chain(
                    q_item["search_chain"], proxy=proxy
                )

                # 检查搜索结果质量
                has_results = any(sr["results"] for sr in search_results)
                if not has_results:
                    print(f"    [FAIL] 所有搜索返回空结果，跳过")
                    continue

                # 基于真实结果合成
                synth = synthesize_response(
                    question, search_results, q_item["num_hops"]
                )
                if not synth:
                    print(f"    [FAIL] 合成失败，跳过")
                    continue

                # 组装完整数据
                final_item = {
                    "question": question,
                    "answer": synth["answer"],
                    "search_chain": q_item["search_chain"],
                    "demo_response": synth["demo_response"],
                    "num_hops": q_item["num_hops"],
                    "difficulty": difficulty,
                    "requires_search": True,
                    "real_search": True,  # 标记使用了真实搜索
                }
                collected.append(final_item)
                print(f"    [OK] ({len(collected)}/{target}) 答案: {synth['answer'][:40]}")

                time.sleep(0.3)

            attempts += 1

        if len(collected) < target:
            print(f"  [WARN]  只生成了 {len(collected)}/{target} 条")

        all_items.extend(collected[:target])

    # 打乱顺序
    random.seed(42)
    random.shuffle(all_items)

    # 写入
    print(f"\n写入 {len(all_items)} 条数据到 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    diff_counts = {}
    hop_counts = {}
    for item in all_items:
        d = item.get("difficulty", "?")
        h = item.get("num_hops", "?")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        hop_counts[h] = hop_counts.get(h, 0) + 1

    print(f"\n[DONE] 成功生成 {len(all_items)} 条基于真实搜索的训练数据！")
    print(f"\n难度分布:")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c} ({c/len(all_items)*100:.1f}%)")
    print(f"\n搜索跳数分布:")
    for h, c in sorted(hop_counts.items()):
        print(f"  {h} 跳: {c} ({c/len(all_items)*100:.1f}%)")

    # 示例
    print("\n示例数据:")
    for i, item in enumerate(all_items[:2], 1):
        print(f"\n  [{i}] 问题: {item['question']}")
        print(f"      答案: {item['answer']}")
        print(f"      跳数: {item['num_hops']}, 难度: {item['difficulty']}")
        print(f"      搜索链:")
        for j, step in enumerate(item["search_chain"], 1):
            print(f"        {j}. {step['query']} ({step['purpose']})")
        demo_preview = item["demo_response"][:200].replace("\n", "\\n")
        print(f"      demo: {demo_preview}...")

    return len(all_items)


def main():
    parser = argparse.ArgumentParser(
        description="生成 Search-R1 多轮中文训练数据 (真实搜索引擎)")
    parser.add_argument("--output", type=str,
                        default="data/search_r1_multiturn_train.jsonl")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP proxy (e.g. http://your-proxy:port)")
    parser.add_argument("--gen-val", action="store_true",
                        help="同时生成验证集")
    parser.add_argument("--val-output", type=str,
                        default="data/search_r1_multiturn_val.jsonl")
    parser.add_argument("--val-examples", type=int, default=50)
    args = parser.parse_args()

    # 测试 API
    print("测试 Gemini API ...")
    test = call_gemini([{"role": "user", "content": "回复'OK'"}],
                       temperature=0, max_tokens=10)
    if test:
        print(f"[OK] API 连接成功: {test.strip()[:30]}\n")
    else:
        print("[FAIL] API 连接失败")
        return 1

    # 生成训练集
    count = generate_dataset(args.output, args.num_examples, args.proxy)
    if count == 0:
        print("\n[FAIL] 训练数据生成失败")
        return 1

    # 生成验证集
    if args.gen_val:
        print("\n" + "=" * 80)
        print("开始生成验证集...")
        print("=" * 80)
        val_count = generate_dataset(args.val_output, args.val_examples, args.proxy)
        if val_count == 0:
            print("\n[WARN]  验证数据生成失败，但训练数据已保存")

    print("\n" + "=" * 80)
    print("下一步：使用真实搜索数据启动训练")
    print("=" * 80)
    proxy_arg = f"    --search-proxy {args.proxy} \\\n" if args.proxy else ""
    print(f"""
python -m scripts.search_r1_grpo \\
    --model-path /path/to/Qwen3-8B \\
    --train-data {args.output} \\
    --val-data {args.val_output if args.gen_val else 'data/search_r1_multiturn_val.jsonl'} \\
    --search-engine duckduckgo \\
{proxy_arg}    --enable-memory \\
    --max-search-turns 5 \\
    --num-epochs 3
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
