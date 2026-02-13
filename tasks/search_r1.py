"""
Search-R1 Task: 多轮交错式搜索推理 (think -> search -> information -> think -> ...)

支持 Search-R1 论文中的多轮 rollout 训练范式：
  模型生成 -> 检测 <search> -> 执行搜索 -> 注入 <information> -> 模型继续生成

数据格式（JSONL 每行）：
  {
    "question": "需要多跳推理的中文问题",
    "answer": "关键词1, 关键词2",
    "search_chain": [{"query": "...", "purpose": "..."}],
    "demo_response": "<think>...</think><search>...</search>...",
    "num_hops": 2,
    "difficulty": "medium",
    "requires_search": true
  }
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.common import Task
from tools.search_tools import SearchToolkit, execute_search_tool
from tools.memory_manager import SearchR1Memory


# ── 系统提示（中文，包含多轮搜索工具说明）──
SYSTEM_PROMPT_CN = """\
你是一个有用的AI助手，可以使用网络搜索工具来获取信息并进行多步推理。

可用工具：
- web_search：搜索网络获取最新信息

回答问题时，请遵循以下流程：
1. 在 <think>...</think> 标签中进行推理思考
2. 需要搜索时，使用 <search>你的搜索查询</search> 标签
3. 搜索结果会以 <information>...</information> 标签返回给你
4. 根据搜索结果继续推理，如果信息不够可以再次搜索
5. 最终在推理完成后直接给出答案

示例格式：
<think>我需要先了解X，然后再查Y</think>
<search>X相关查询</search>
<information>搜索结果...</information>
<think>根据搜索结果，我了解到...但还需要进一步了解Y</think>
<search>Y相关查询</search>
<information>搜索结果...</information>
<think>综合以上信息，我可以得出结论</think>
最终答案"""


@dataclass
class SearchR1Example:
    """Search-R1 多轮训练样例"""
    question: str
    ground_truth: str                      # 答案关键词
    search_chain: List[Dict] = field(default_factory=list)  # 预期搜索链
    demo_response: str = ""                # 示范多轮响应
    num_hops: int = 1                      # 预期搜索跳数
    difficulty: str = "medium"
    requires_search: bool = True
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    def to_messages(self, include_history: bool = True) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT_CN}]
        if include_history and self.conversation_history:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": self.question})
        return messages


class SearchR1Task(Task):
    """Search-R1 多轮 RL 任务"""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        search_engine: str = "mock",
        enable_memory: bool = True,
        max_search_turns: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.enable_memory = enable_memory
        self.max_search_turns = max_search_turns

        self.examples = self._load_data(data_path)
        print(f"Loaded {len(self.examples)} examples from {data_path}")

        self.search_toolkit = SearchToolkit(
            preferred_engines=[search_engine, "mock"])
        self.memories: Dict[int, SearchR1Memory] = {}

    # ──────────────────────── 数据加载 ────────────────────────
    def _load_data(self, data_path: str) -> List[SearchR1Example]:
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, using sample data")
            return self._generate_sample_data()

        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                examples.append(SearchR1Example(
                    question=d["question"],
                    ground_truth=d.get("answer", ""),
                    search_chain=d.get("search_chain", []),
                    demo_response=d.get("demo_response", ""),
                    num_hops=d.get("num_hops", 1),
                    difficulty=d.get("difficulty", "medium"),
                    requires_search=d.get("requires_search", True),
                    conversation_history=d.get("history", []),
                ))
        return examples

    def _generate_sample_data(self) -> List[SearchR1Example]:
        return [
            SearchR1Example(
                question="2024年诺贝尔物理学奖得主之一杰弗里·辛顿，他在哪所大学任教？该大学的现任校长是谁？",
                ground_truth="多伦多大学, Meric Gertler",
                search_chain=[
                    {"query": "杰弗里·辛顿 任教大学", "purpose": "查找辛顿的任教单位"},
                    {"query": "多伦多大学 现任校长", "purpose": "查找大学校长"},
                ],
                demo_response=(
                    "<think>这个问题有两层：先要找辛顿的大学，再找大学校长。让我分步搜索。</think>\n"
                    "<search>杰弗里·辛顿 任教大学</search>\n"
                    "<information>杰弗里·辛顿（Geoffrey Hinton）是多伦多大学的名誉教授。</information>\n"
                    "<think>好的，辛顿在多伦多大学任教。现在需要搜索多伦多大学现任校长。</think>\n"
                    "<search>多伦多大学 现任校长 2024</search>\n"
                    "<information>多伦多大学现任校长是Meric Gertler，自2013年起就任。</information>\n"
                    "<think>综合以上信息：辛顿在多伦多大学任教，现任校长是Meric Gertler。</think>\n"
                    "杰弗里·辛顿在多伦多大学任教，该大学的现任校长是Meric Gertler。"
                ),
                num_hops=2,
                difficulty="medium",
            ),
            SearchR1Example(
                question="全球芯片短缺最初由什么事件引发？这对中国汽车产业有什么影响？中国采取了哪些应对措施？",
                ground_truth="新冠疫情, 汽车减产, 芯片自主, 国产替代",
                search_chain=[
                    {"query": "全球芯片短缺原因", "purpose": "查找芯片短缺起因"},
                    {"query": "芯片短缺 中国汽车产业 影响", "purpose": "了解对中国汽车的影响"},
                    {"query": "中国 芯片短缺 应对措施 政策", "purpose": "查找应对政策"},
                ],
                demo_response=(
                    "<think>这个问题需要三步：1)芯片短缺原因 2)对中国汽车影响 3)中国应对措施。</think>\n"
                    "<search>全球芯片短缺 原因 起因</search>\n"
                    "<information>全球芯片短缺始于2020年新冠疫情，导致半导体工厂停工，加上居家办公推动电子产品需求激增。</information>\n"
                    "<think>芯片短缺由新冠疫情引发。接下来了解对中国汽车的具体影响。</think>\n"
                    "<search>芯片短缺 中国汽车产业 影响 2021 2022</search>\n"
                    "<information>芯片短缺导致中国多家车企减产，2021年中国汽车产量因此减少约200万辆。</information>\n"
                    "<think>影响很大，导致大规模减产。最后搜索中国采取的应对措施。</think>\n"
                    "<search>中国 芯片 自主 国产替代 政策</search>\n"
                    "<information>中国加大芯片自主研发投入，推动国产替代，设立集成电路产业投资基金。</information>\n"
                    "<think>综合分析完毕。</think>\n"
                    "全球芯片短缺最初由2020年新冠疫情引发，这导致中国汽车产业大规模减产，"
                    "中国采取了加大芯片自主研发、推动国产替代等应对措施。"
                ),
                num_hops=3,
                difficulty="hard",
            ),
        ]

    # ──────────────────────── Task 接口 ────────────────────────
    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]

        # 记忆
        if self.enable_memory:
            if index not in self.memories:
                self.memories[index] = SearchR1Memory(max_turns=5)
            memory = self.memories[index]
        else:
            memory = None

        messages = example.to_messages(include_history=True)

        # 追加记忆历史
        if memory and memory.conversation.turn_count > 0:
            memory_msgs = memory.get_context_messages()[1:]
            messages = messages[:1] + memory_msgs + messages[1:]

        return {
            "messages": messages,
            "ground_truth": example.ground_truth,
            "requires_search": example.requires_search,
            "difficulty": example.difficulty,
            "num_hops": example.num_hops,
            "search_chain": [s.get("query", "") for s in example.search_chain],
            "demo_response": example.demo_response,
            "example_id": index,
        }

    # ──────────────────────── 工具解析与执行 ────────────────────────
    @staticmethod
    def parse_search_tags(text: str) -> List[str]:
        """从文本中提取所有 <search>query</search> 查询"""
        return re.findall(r"<search>(.*?)</search>", text, re.DOTALL)

    @staticmethod
    def count_think_tags(text: str) -> int:
        """统计 <think>...</think> 块的数量"""
        return len(re.findall(r"<think>.*?</think>", text, re.DOTALL))

    def execute_search(self, query: str) -> str:
        """执行一次搜索并返回结果文本"""
        call = {"name": "web_search", "arguments": {"query": query.strip()}}
        result = execute_search_tool(self.search_toolkit, call)
        # 截断过长的搜索结果
        if isinstance(result, str) and len(result) > 500:
            result = result[:500] + "..."
        return str(result)

    # ──────────────────────── 多轮 Rollout ────────────────────────
    def multiturn_rollout(
        self,
        generate_fn,       # (token_ids: List[int]) -> List[int]  单次生成函数
        prompt_tokens: List[int],
        tokenizer,
        max_new_tokens: int = 512,
        max_turns: int = 5,
    ) -> str:
        """
        执行多轮 rollout：
          1. 模型从 prompt 开始生成
          2. 检测 </search> -> 截断生成，执行搜索
          3. 追加 <information>结果</information> 到上下文
          4. 模型从新上下文继续生成
          5. 重复直到无更多搜索或达到 max_turns

        返回完整的响应文本（包含所有 think/search/information 块）
        """
        full_context_ids = list(prompt_tokens)
        full_response_parts = []
        search_tag_end = "</search>"
        turns_used = 0

        for turn in range(max_turns):
            # 生成一段文本
            new_ids = generate_fn(full_context_ids)
            new_text = tokenizer.decode(new_ids)

            # 检查是否包含 </search>
            if search_tag_end in new_text:
                # 截断到 </search> 之后
                idx = new_text.index(search_tag_end) + len(search_tag_end)
                new_text_truncated = new_text[:idx]
                full_response_parts.append(new_text_truncated)

                # 提取搜索查询
                queries = self.parse_search_tags(new_text_truncated)
                if queries:
                    query = queries[-1]  # 最后一个搜索查询
                    search_result = self.execute_search(query)
                    info_block = f"\n<information>{search_result}</information>\n"
                    full_response_parts.append(info_block)

                    # 更新上下文
                    continued_text = new_text_truncated + info_block
                    continued_ids = tokenizer.encode(continued_text)
                    full_context_ids = list(prompt_tokens) + continued_ids
                    turns_used += 1
                else:
                    # 有 </search> 但没提取到查询，停止
                    break
            else:
                # 没有搜索标签 -> 最终生成，结束
                full_response_parts.append(new_text)
                break

        full_response = "".join(full_response_parts)
        return full_response

    # ──────────────────────── 奖励函数 ────────────────────────
    def reward(self, item: Dict[str, Any], assistant_response: str) -> float:
        """
        多轮搜索奖励函数

        5 个维度：
          1. 答案正确性     (0 ~ 1.0)   — 最重要
          2. 搜索行为质量   (0 ~ 0.3)   — 是否正确使用多轮搜索
          3. 推理质量       (0 ~ 0.2)   — think 标签中的推理
          4. 格式规范       (0 ~ 0.1)   — 标签格式是否正确
          5. 效率惩罚       (-0.1)      — 过多冗余搜索扣分

        总分归一化到 [0, 1]
        """
        total = 0.0
        breakdown = {}

        # ── 1. 答案正确性 ──
        gt_lower = item["ground_truth"].lower()
        resp_lower = assistant_response.lower()

        cn_stopwords = {
            "的", "了", "和", "是", "在", "有", "我", "他", "她", "它",
            "这", "那", "就", "也", "都", "会", "对", "到", "说", "要",
            "把", "让", "被", "从", "而", "但", "又", "与", "及", "等",
        }

        gt_variants = [v.strip() for v in
                       gt_lower.replace("/", ",").replace("、", ",").split(",")]
        gt_keywords = set()
        for v in gt_variants:
            words = v.split()
            gt_keywords.update(words)
            if not any(c == " " for c in v) and any(
                    "\u4e00" <= c <= "\u9fff" for c in v):
                gt_keywords.add(v)
        gt_keywords -= cn_stopwords
        gt_keywords = {k for k in gt_keywords if len(k) >= 2 or k.isascii()}

        if gt_keywords:
            matched = sum(1 for k in gt_keywords if k in resp_lower)
            correctness = matched / len(gt_keywords)
        else:
            correctness = 0.5

        if any(v in resp_lower for v in gt_variants if v):
            correctness = max(correctness, 0.9)
        if gt_lower in resp_lower:
            correctness = max(correctness, 0.95)

        breakdown["correctness"] = correctness
        total += correctness

        # ── 2. 搜索行为质量 ──
        search_queries = self.parse_search_tags(assistant_response)
        num_searches = len(search_queries)
        expected_hops = item.get("num_hops", 1)
        expected_chain = item.get("search_chain", [])

        search_reward = 0.0

        if item.get("requires_search", True):
            # (a) 是否使用了搜索
            if num_searches > 0:
                search_reward += 0.1

            # (b) 搜索次数是否接近预期跳数
            if num_searches >= expected_hops:
                search_reward += 0.1
            elif num_searches > 0:
                search_reward += 0.05  # 至少搜了，但不够

            # (c) 搜索查询与预期搜索链的相关性
            if expected_chain and search_queries:
                chain_overlap = 0
                for expected_q in expected_chain:
                    eq_chars = set(expected_q) - cn_stopwords - set(" ?？。，")
                    for actual_q in search_queries:
                        aq_chars = set(actual_q) - cn_stopwords - set(" ?？。，")
                        if eq_chars and aq_chars:
                            overlap = len(eq_chars & aq_chars) / max(
                                len(eq_chars), 1)
                            if overlap > 0.3:
                                chain_overlap += 1
                                break
                if len(expected_chain) > 0:
                    search_reward += 0.1 * (chain_overlap / len(expected_chain))

            # (d) 惩罚过多冗余搜索
            if num_searches > expected_hops + 2:
                search_reward -= 0.1
        else:
            if num_searches > 0:
                search_reward -= 0.05

        search_reward = max(0.0, min(0.3, search_reward))
        breakdown["search_behavior"] = search_reward
        total += search_reward

        # ── 3. 推理质量（think 标签内容）──
        reasoning_reward = 0.0
        think_blocks = re.findall(
            r"<think>(.*?)</think>", assistant_response, re.DOTALL)

        # (a) 有 think 标签
        if think_blocks:
            reasoning_reward += 0.05

        # (b) 多轮思考（有多个 think 块）
        if len(think_blocks) >= 2:
            reasoning_reward += 0.05

        # (c) 反思性短语
        reflective_phrases = [
            "让我", "我来", "需要进一步", "根据搜索结果", "综合", "分析",
            "可以看出", "由此可见", "首先", "其次", "然后", "因此",
            "重新考虑", "进一步搜索", "还需要", "接下来",
            "根据以上", "总结来看", "通过搜索",
        ]
        phrase_count = sum(
            1 for p in reflective_phrases if p in assistant_response)
        reasoning_reward += min(0.1, phrase_count * 0.02)

        reasoning_reward = min(0.2, reasoning_reward)
        breakdown["reasoning"] = reasoning_reward
        total += reasoning_reward

        # ── 4. 格式规范 ──
        format_reward = 0.0

        # 正确使用了 think + search 标签对
        has_think = "<think>" in assistant_response and "</think>" in assistant_response
        has_search = "<search>" in assistant_response and "</search>" in assistant_response
        if has_think:
            format_reward += 0.03
        if has_search:
            format_reward += 0.03

        # 标签嵌套正确（think 在 search 之前）
        first_think = assistant_response.find("<think>")
        first_search = assistant_response.find("<search>")
        if first_think >= 0 and first_search >= 0 and first_think < first_search:
            format_reward += 0.02

        # 最终答案不在 think/search 标签内
        last_think_end = assistant_response.rfind("</think>")
        if last_think_end > 0 and len(assistant_response) > last_think_end + 10:
            # 有标签之后的最终答案文本
            format_reward += 0.02

        format_reward = min(0.1, format_reward)
        breakdown["format"] = format_reward
        total += format_reward

        # ── 5. 难度系数 ──
        difficulty_mult = {"easy": 1.0, "medium": 1.1, "hard": 1.2}.get(
            item.get("difficulty", "medium"), 1.0)
        total *= difficulty_mult

        # 归一化到 [0, 1]
        total = max(0.0, min(1.0, total / 1.6))

        if os.getenv("DEBUG_REWARD"):
            print(f"\n=== Reward Breakdown ===")
            for k, v in breakdown.items():
                print(f"  {k}: {v:.3f}")
            print(f"  difficulty_mult: {difficulty_mult}")
            print(f"  TOTAL: {total:.3f}")

        return total

    def evaluate(self, item: Dict[str, Any], assistant_response: str) -> int:
        return 1 if self.reward(item, assistant_response) > 0.5 else 0

    def update_memory(self, example_id: int, assistant_response: str):
        if not self.enable_memory or example_id not in self.memories:
            return
        memory = self.memories[example_id]
        example = self.examples[example_id]
        memory.add_user_message(example.question)

        queries = self.parse_search_tags(assistant_response)
        tool_results = []
        for q in queries:
            result = self.execute_search(q)
            tool_results.append({
                "name": "web_search",
                "arguments": {"query": q},
                "result": result,
            })
        memory.add_assistant_message(assistant_response, tool_calls=tool_results)


def generate_search_r1_dataset(
    output_path: str,
    num_examples: int = 100,
    difficulty_mix: Dict[str, float] = None,
):
    """生成兼容格式的示例数据（无 Gemini，用于快速测试）"""
    if difficulty_mix is None:
        difficulty_mix = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

    examples = []

    easy_questions = [
        {
            "question": "世界上最深的湖泊在哪个国家？该国首都是哪里？",
            "answer": "俄罗斯, 贝加尔湖, 莫斯科",
            "search_chain": [
                {"query": "世界上最深的湖泊", "purpose": "找到最深湖泊"},
                {"query": "俄罗斯首都", "purpose": "找到首都"},
            ],
            "num_hops": 2,
        },
        {
            "question": "中国最长的河流叫什么？它流经哪些主要城市？",
            "answer": "长江, 武汉, 南京, 上海",
            "search_chain": [
                {"query": "中国最长的河流", "purpose": "找到河流"},
                {"query": "长江 流经 城市", "purpose": "查找流经城市"},
            ],
            "num_hops": 2,
        },
    ]

    medium_questions = [
        {
            "question": "OpenAI的CEO是谁？他之前创办过什么公司？",
            "answer": "Sam Altman, Y Combinator, Loopt",
            "search_chain": [
                {"query": "OpenAI CEO", "purpose": "找到CEO"},
                {"query": "Sam Altman 创办 公司 经历", "purpose": "查找创业历史"},
            ],
            "num_hops": 2,
        },
    ]

    hard_questions = [
        {
            "question": "量子计算的基本原理是什么？谷歌在量子计算方面的最新突破是什么？这对密码学有什么影响？",
            "answer": "量子比特, 叠加态, Willow, 量子霸权, RSA",
            "search_chain": [
                {"query": "量子计算 基本原理", "purpose": "理解原理"},
                {"query": "谷歌 量子计算 最新突破 Willow", "purpose": "最新进展"},
                {"query": "量子计算 密码学 影响 RSA", "purpose": "安全影响"},
            ],
            "num_hops": 3,
        },
    ]

    all_templates = {
        "easy": easy_questions,
        "medium": medium_questions,
        "hard": hard_questions,
    }

    for difficulty, ratio in difficulty_mix.items():
        n = int(num_examples * ratio)
        templates = all_templates.get(difficulty, medium_questions)
        for i in range(n):
            t = templates[i % len(templates)]
            examples.append({
                **t,
                "difficulty": difficulty,
                "requires_search": True,
                "demo_response": "",
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    print("Testing SearchR1Task (multi-turn)...")

    data_path = "/tmp/search_r1_multiturn_test.jsonl"
    generate_search_r1_dataset(data_path, num_examples=10)

    task = SearchR1Task(data_path, search_engine="mock", enable_memory=True)

    item = task[0]
    print(f"\nQuestion: {item['messages'][-1]['content']}")
    print(f"Ground truth: {item['ground_truth']}")
    print(f"Num hops: {item['num_hops']}")
    print(f"Search chain: {item['search_chain']}")

    # 测试多轮响应
    good_response = (
        "<think>我需要先找世界上最深的湖泊，然后查它所在国家的首都。</think>\n"
        "<search>世界上最深的湖泊</search>\n"
        "<information>世界上最深的湖泊是贝加尔湖，位于俄罗斯西伯利亚。</information>\n"
        "<think>好的，最深湖泊是俄罗斯的贝加尔湖。现在搜索俄罗斯的首都。</think>\n"
        "<search>俄罗斯首都</search>\n"
        "<information>俄罗斯的首都是莫斯科。</information>\n"
        "<think>综合以上信息，我可以回答了。</think>\n"
        "世界上最深的湖泊是贝加尔湖，位于俄罗斯。俄罗斯的首都是莫斯科。"
    )

    bad_response = "我不确定。"

    os.environ["DEBUG_REWARD"] = "1"
    print(f"\n=== Good Response ===")
    r_good = task.reward(item, good_response)
    print(f"Reward: {r_good:.3f}")

    print(f"\n=== Bad Response ===")
    r_bad = task.reward(item, bad_response)
    print(f"Reward: {r_bad:.3f}")

    assert r_good > r_bad, f"Good ({r_good}) should beat bad ({r_bad})"
    print(f"\n[OK] Good ({r_good:.3f}) > Bad ({r_bad:.3f})")
    print("[OK] All tests passed!")
