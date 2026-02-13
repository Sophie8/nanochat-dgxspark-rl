#!/usr/bin/env python3
"""
使用 Gemini 3 Flash Preview 生成 Search-R1 训练数据

Usage:
    python scripts/generate_search_r1_data_with_gemini.py \
        --output data/search_r1_train_gemini.jsonl \
        --num-examples 200
"""

import argparse
import json
import os
import sys
from typing import List, Dict
import requests
from tqdm import tqdm
import time

# Gemini API 配置 (通过环境变量设置)
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def call_gemini(messages: List[Dict], temperature: float = 0.8, max_tokens: int = 500) -> str:
    """调用 Gemini API"""
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None


# 问题类型模板
QUESTION_TYPES = [
    {
        "category": "时事新闻",
        "difficulty": "medium",
        "examples": [
            "最近有哪些重要的科技新闻？",
            "2024年最新的AI突破是什么？",
            "当前全球经济形势如何？"
        ]
    },
    {
        "category": "科学技术",
        "difficulty": "hard",
        "examples": [
            "量子计算的最新进展是什么？",
            "CRISPR基因编辑技术有哪些新应用？",
            "核聚变能源研究现状如何？"
        ]
    },
    {
        "category": "数据查询",
        "difficulty": "easy",
        "examples": [
            "中国的人口数量是多少？",
            "世界上最高的山是什么？",
            "比特币现在的价格是多少？"
        ]
    },
    {
        "category": "比较分析",
        "difficulty": "hard",
        "examples": [
            "Python和JavaScript哪个更适合初学者？",
            "特斯拉和比亚迪的电动车有什么区别？",
            "ChatGPT和Gemini的能力对比如何？"
        ]
    },
    {
        "category": "解释说明",
        "difficulty": "medium",
        "examples": [
            "什么是区块链技术？",
            "全球变暖的主要原因是什么？",
            "5G和4G有什么区别？"
        ]
    },
    {
        "category": "历史事件",
        "difficulty": "easy",
        "examples": [
            "第一次世界大战是什么时候发生的？",
            "中国的首都是哪里？",
            "互联网是什么时候发明的？"
        ]
    },
    {
        "category": "健康医疗",
        "difficulty": "medium",
        "examples": [
            "COVID-19疫苗有哪些类型？",
            "如何预防心血管疾病？",
            "糖尿病的症状有哪些？"
        ]
    },
    {
        "category": "商业金融",
        "difficulty": "hard",
        "examples": [
            "什么是量化宽松政策？",
            "股票市场的牛市和熊市是什么意思？",
            "加密货币的未来前景如何？"
        ]
    }
]


def generate_question_batch(category: str, difficulty: str, batch_size: int = 10) -> List[Dict]:
    """使用 Gemini 生成一批问题"""
    
    prompt = f"""请生成 {batch_size} 个关于"{category}"的问题，这些问题应该：

1. 需要通过网络搜索才能获得准确答案
2. 难度级别：{difficulty}
3. 问题应该多样化，涵盖不同角度
4. 每个问题都应该清晰、具体
5. 适合用来训练AI助手使用搜索工具

请以JSON格式返回，每个问题包含：
- question: 问题内容
- answer_keywords: 答案的关键词（2-5个，用于评估）
- requires_search: true（都需要搜索）
- difficulty: "{difficulty}"

返回格式：
{{"questions": [
    {{"question": "...", "answer_keywords": "关键词1, 关键词2", "requires_search": true, "difficulty": "{difficulty}"}},
    ...
]}}

只返回JSON，不要其他文字。"""
    
    messages = [
        {"role": "system", "content": "你是一个专业的数据生成助手，擅长创建高质量的训练数据。"},
        {"role": "user", "content": prompt}
    ]
    
    response = call_gemini(messages, temperature=0.9, max_tokens=1000)
    
    if not response:
        return []
    
    try:
        # 尝试解析JSON
        # 有时候模型会在JSON前后添加文字，需要提取
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response)
        questions = data.get("questions", [])
        
        # 验证和清理数据
        valid_questions = []
        for q in questions:
            if "question" in q and "answer_keywords" in q:
                valid_questions.append({
                    "question": q["question"],
                    "answer": q["answer_keywords"],  # 使用 answer 字段
                    "requires_search": q.get("requires_search", True),
                    "difficulty": difficulty
                })
        
        return valid_questions
    
    except Exception as e:
        print(f"解析失败: {e}")
        print(f"响应内容: {response[:200]}...")
        return []


def generate_dataset(output_path: str, num_examples: int = 200):
    """生成完整的训练数据集"""
    
    print("=" * 80)
    print(f"使用 Gemini 3 Flash Preview 生成 Search-R1 训练数据")
    print(f"目标数量: {num_examples}")
    print(f"输出路径: {output_path}")
    print("=" * 80)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    all_questions = []
    
    # 计算每个类别应该生成多少问题
    num_categories = len(QUESTION_TYPES)
    questions_per_category = num_examples // num_categories
    remaining = num_examples % num_categories
    
    print(f"\n共 {num_categories} 个类别，每个类别生成 {questions_per_category} 个问题")
    print()
    
    for idx, qtype in enumerate(QUESTION_TYPES):
        category = qtype["category"]
        difficulty = qtype["difficulty"]
        
        # 最后一个类别补充剩余数量
        target_count = questions_per_category
        if idx == num_categories - 1:
            target_count += remaining
        
        print(f"[{idx+1}/{num_categories}] 生成类别: {category} (难度: {difficulty}, 目标: {target_count})")
        
        category_questions = []
        attempts = 0
        max_attempts = 10
        
        while len(category_questions) < target_count and attempts < max_attempts:
            batch_size = min(10, target_count - len(category_questions))
            print(f"  尝试 {attempts+1}/{max_attempts}, 已生成 {len(category_questions)}/{target_count}...")
            
            questions = generate_question_batch(category, difficulty, batch_size)
            
            if questions:
                category_questions.extend(questions)
                print(f"  [OK] 成功生成 {len(questions)} 个问题")
            else:
                print(f"  [FAIL] 生成失败，重试...")
            
            attempts += 1
            time.sleep(1)  # 避免API限流
        
        if len(category_questions) < target_count:
            print(f"  [WARN]  警告: 只生成了 {len(category_questions)}/{target_count} 个问题")
        
        all_questions.extend(category_questions[:target_count])
        print()
    
    # 写入文件
    print(f"写入数据到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    print(f"\n[DONE] 成功生成 {len(all_questions)} 条训练数据！")
    print(f"文件保存在: {output_path}")
    
    # 统计信息
    difficulty_counts = {}
    for q in all_questions:
        diff = q.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print("\n数据统计:")
    print(f"  总数: {len(all_questions)}")
    print(f"  难度分布:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"    {diff}: {count} ({count/len(all_questions)*100:.1f}%)")
    
    # 显示示例
    print("\n示例数据:")
    for i, q in enumerate(all_questions[:3], 1):
        print(f"\n  [{i}] {q['question']}")
        print(f"      答案关键词: {q['answer']}")
        print(f"      难度: {q['difficulty']}")
    
    return len(all_questions)


def main():
    parser = argparse.ArgumentParser(description="使用 Gemini 生成 Search-R1 训练数据")
    parser.add_argument("--output", type=str, default="data/search_r1_train_gemini.jsonl",
                        help="输出文件路径")
    parser.add_argument("--num-examples", type=int, default=200,
                        help="生成的样例数量")
    args = parser.parse_args()
    
    # 测试API连接
    print("测试 Gemini API 连接...")
    test_response = call_gemini([
        {"role": "user", "content": "你好，请回复'连接成功'"}
    ])
    
    if test_response:
        print(f"[OK] API 连接成功: {test_response[:50]}...")
        print()
    else:
        print("[FAIL] API 连接失败，请检查配置")
        return 1
    
    # 生成数据集
    count = generate_dataset(args.output, args.num_examples)
    
    if count > 0:
        print("\n" + "=" * 80)
        print("下一步：开始训练")
        print("=" * 80)
        print(f"""
# 单GPU训练
python -m scripts.search_r1_grpo \\
    --model-path /path/to/Qwen3-8B \\
    --train-data {args.output} \\
    --search-engine tavily \\
    --enable-memory \\
    --num-epochs 3

# 双节点训练
MASTER_ADDR=<ip> NODE_RANK=0 \\
TRAIN_DATA={args.output} \\
bash scripts/run_search_r1_2node.sh
""")
        return 0
    else:
        print("\n[FAIL] 数据生成失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

