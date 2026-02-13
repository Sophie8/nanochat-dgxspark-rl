#!/usr/bin/env python3
"""用 Gemini 搜索引擎生成验证数据（DuckDuckGo 代理不可用时的备选方案）"""

import json, sys, os, time, random, requests

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def call_gemini(messages, temperature=0.8, max_tokens=2000, retries=3):
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GEMINI_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(retries):
        try:
            resp = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=45)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API失败({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def gemini_search(query, num_results=3):
    prompt = (
        f"你是一个搜索引擎。请根据查询返回{num_results}条真实准确的搜索结果摘要。\n"
        f"查询：{query}\n"
        f'请以JSON返回：{{"results": [{{"title": "...", "url": "https://...", "snippet": "..."}}]}}'
    )
    resp = call_gemini(
        [{"role": "system", "content": "你是搜索引擎，返回准确的JSON结果"},
         {"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=800
    )
    if not resp:
        return []
    try:
        text = resp.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
            for r in data.get("results", [])
        ]
    except:
        return []


def format_results(results):
    if not results:
        return "未找到结果"
    return "\n".join(f'[{i+1}] {r["title"]}\n{r["snippet"][:200]}' for i, r in enumerate(results))


CATS = [
    {"category": "人物关联", "difficulty": "medium", "num_hops": 2,
     "description": "先搜人物再搜关联信息",
     "examples": ["《三体》作者刘慈欣是哪里人？家乡有什么著名景点？"]},
    {"category": "因果链推理", "difficulty": "hard", "num_hops": 3,
     "description": "逐步搜索因果链条",
     "examples": ["芯片短缺如何影响中国汽车产业？中国如何应对？"]},
    {"category": "比较多源", "difficulty": "hard", "num_hops": 2,
     "description": "分别搜索两个实体进行对比",
     "examples": ["中国高铁和日本新干线在时速方面有何不同？"]},
    {"category": "时间线追踪", "difficulty": "medium", "num_hops": 2,
     "description": "搜索事件发展时间线",
     "examples": ["中国空间站从立项到建成的关键节点？"]},
    {"category": "概念与实例", "difficulty": "easy", "num_hops": 2,
     "description": "先搜概念再搜实例",
     "examples": ["什么是碳中和？中国有哪些成就？"]},
    {"category": "地理文化关联", "difficulty": "easy", "num_hops": 2,
     "description": "先搜地理实体再搜深层信息",
     "examples": ["世界最深湖泊在哪国？该国首都是哪？"]},
    {"category": "科技前沿追踪", "difficulty": "hard", "num_hops": 3,
     "description": "追踪前沿科技原理与进展",
     "examples": ["量子纠错基本原理是什么？最新突破有哪些？"]},
    {"category": "政策影响分析", "difficulty": "medium", "num_hops": 2,
     "description": "先搜政策再搜影响",
     "examples": ["欧盟AI法案规定了什么？对中国有何影响？"]},
]

TARGET = 50
OUTPUT = "data/search_r1_multiturn_val.jsonl"


def main():
    print("=" * 60)
    print("验证数据生成 (Gemini 搜索引擎)")
    print(f"目标: {TARGET} 条 -> {OUTPUT}")
    print("=" * 60)

    # 测试 API
    test = call_gemini([{"role": "user", "content": "回复OK"}], temperature=0, max_tokens=10)
    if not test:
        print("Gemini API 不可用!")
        return 1
    print(f"[OK] Gemini API 正常\n")

    per_cat = TARGET // len(CATS)
    remainder = TARGET % len(CATS)
    all_items = []

    for idx, cat in enumerate(CATS):
        target = per_cat + (1 if idx < remainder else 0)
        print(f"\n[{idx+1}/{len(CATS)}] {cat['category']} 目标:{target}")

        collected = []
        for attempt in range(10):
            if len(collected) >= target:
                break
            need = min(5, target - len(collected))

            # 第一步: 生成问题
            prompt = (
                f"生成{need}条需要{cat['num_hops']}跳搜索的中文问题。\n"
                f"类别: {cat['category']}\n说明: {cat['description']}\n"
                f"参考: {cat['examples'][0]}\n\n"
                f'返回JSON: {{"data":[{{"question":"...","search_chain":[{{"query":"...","purpose":"..."}}]}}]}}'
            )
            resp = call_gemini(
                [{"role": "system", "content": "生成多跳搜索问题，只返回JSON"},
                 {"role": "user", "content": prompt}],
                temperature=0.9, max_tokens=3000
            )
            if not resp:
                continue

            try:
                text = resp.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                questions = json.loads(text).get("data", [])
            except:
                continue

            for q in questions:
                if len(collected) >= target:
                    break
                if "question" not in q or "search_chain" not in q:
                    continue

                # 第二步: Gemini 搜索
                search_results_list = []
                for step in q["search_chain"]:
                    results = gemini_search(step["query"])
                    formatted = format_results(results)
                    search_results_list.append({
                        "query": step["query"],
                        "purpose": step.get("purpose", ""),
                        "formatted": formatted,
                    })
                    time.sleep(0.3)

                # 第三步: 合成
                search_info = ""
                for i, sr in enumerate(search_results_list, 1):
                    search_info += f'\n--- 第{i}次搜索 ---\n查询: {sr["query"]}\n结果:\n{sr["formatted"]}\n'

                synth_prompt = (
                    f"基于问题和搜索结果生成答案和多轮推理响应。\n\n"
                    f"问题: {q['question']}\n\n搜索结果: {search_info}\n\n"
                    f"要求:\n"
                    f"1. answer: 最终答案关键词(2-5个,逗号分隔)\n"
                    f"2. demo_response: 使用<think>...<search>...<information>...标签的多轮推理\n\n"
                    f'返回JSON: {{"answer":"关键词","demo_response":"<think>...</think>\\n<search>...</search>\\n<information>...</information>\\n..."}}'
                )
                synth_resp = call_gemini(
                    [{"role": "system", "content": "根据搜索结果合成回答，只返回JSON"},
                     {"role": "user", "content": synth_prompt}],
                    temperature=0.5, max_tokens=2500
                )
                if not synth_resp:
                    continue

                try:
                    st = synth_resp.strip()
                    if "```json" in st:
                        st = st.split("```json")[1].split("```")[0].strip()
                    elif "```" in st:
                        st = st.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(st)
                    demo = parsed.get("demo_response", "")
                    answer = parsed.get("answer", "")
                    if "<think>" in demo and "<search>" in demo and answer:
                        collected.append({
                            "question": q["question"],
                            "answer": answer,
                            "search_chain": q["search_chain"],
                            "demo_response": demo,
                            "num_hops": cat["num_hops"],
                            "difficulty": cat["difficulty"],
                            "requires_search": True,
                            "real_search": False,
                            "search_engine": "gemini",
                        })
                        print(f"  [OK] ({len(collected)}/{target}) {answer[:40]}")
                except:
                    continue

            time.sleep(0.5)

        if len(collected) < target:
            print(f"  [WARN] 只有 {len(collected)}/{target}")
        all_items.extend(collected[:target])

    random.seed(123)
    random.shuffle(all_items)

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] 验证数据生成完成: {len(all_items)} 条 -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

