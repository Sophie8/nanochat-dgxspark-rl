"""
Memory Manager for Multi-turn Conversations
支持对话历史管理、上下文压缩和记忆检索
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    role: str  # 'user' or 'assistant' or 'system'
    content: str
    timestamp: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_turns: int = 10, max_tokens: int = 2048):
        """
        Args:
            max_turns: 保留的最大对话轮数
            max_tokens: 上下文的最大 token 数（近似）
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.memory: deque = deque(maxlen=max_turns * 2)  # user + assistant = 2
        self.turn_count = 0
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加一条消息到记忆"""
        entry = MemoryEntry(
            role=role,
            content=content,
            timestamp=self.turn_count,
            metadata=metadata or {}
        )
        self.memory.append(entry)
        
        if role == "user":
            self.turn_count += 1
    
    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        获取消息列表（用于模型输入）
        
        Returns:
            [{"role": "user", "content": "..."}, ...]
        """
        messages = []
        for entry in self.memory:
            if not include_system and entry.role == "system":
                continue
            messages.append({
                "role": entry.role,
                "content": entry.content
            })
        return messages
    
    def get_recent_turns(self, n: int = 3) -> List[Dict[str, str]]:
        """获取最近 N 轮对话"""
        recent = list(self.memory)[-n*2:]  # user + assistant
        return [{"role": e.role, "content": e.content} for e in recent]
    
    def compress(self, tokenizer=None):
        """
        压缩记忆：保留最重要的对话
        策略：保留最近的对话，丢弃中间的
        """
        if len(self.memory) <= self.max_turns * 2:
            return
        
        # 简单策略：保留第一条（通常是 system）和最近的对话
        first_message = self.memory[0] if self.memory else None
        recent_messages = list(self.memory)[-(self.max_turns * 2 - 1):]
        
        self.memory.clear()
        if first_message:
            self.memory.append(first_message)
        self.memory.extend(recent_messages)
    
    def clear(self):
        """清空记忆"""
        self.memory.clear()
        self.turn_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "turn_count": self.turn_count,
            "memory": [e.to_dict() for e in self.memory]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """从字典反序列化"""
        memory_manager = cls(
            max_turns=data.get("max_turns", 10),
            max_tokens=data.get("max_tokens", 2048)
        )
        memory_manager.turn_count = data.get("turn_count", 0)
        
        for entry_dict in data.get("memory", []):
            entry = MemoryEntry(**entry_dict)
            memory_manager.memory.append(entry)
        
        return memory_manager


class ToolCallMemory:
    """工具调用记忆管理器"""
    
    def __init__(self, max_calls: int = 20):
        """
        Args:
            max_calls: 保留的最大工具调用数
        """
        self.max_calls = max_calls
        self.calls: deque = deque(maxlen=max_calls)
    
    def add_call(self, tool_name: str, arguments: Dict[str, Any], result: str):
        """记录工具调用"""
        call_record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": len(self.calls)
        }
        self.calls.append(call_record)
    
    def get_recent_calls(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的工具调用"""
        return list(self.calls)[-n:]
    
    def has_recent_search(self, query: str, similarity_threshold: float = 0.5) -> bool:
        """
        检查是否最近搜索过类似的内容
        
        Args:
            query: 当前查询
            similarity_threshold: 相似度阈值（默认 0.5）
            
        Returns:
            是否有相似的最近搜索
        """
        # 简单的字符串匹配（可以升级为语义相似度）
        for call in list(self.calls)[-5:]:
            if call["tool_name"] == "web_search":
                prev_query = call["arguments"].get("query", "")
                # 简单的 Jaccard 相似度
                words1 = set(query.lower().split())
                words2 = set(prev_query.lower().split())
                if not words1 or not words2:
                    continue
                similarity = len(words1 & words2) / len(words1 | words2)
                if similarity >= similarity_threshold:
                    return True
        return False
    
    def clear(self):
        """清空记忆"""
        self.calls.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "max_calls": self.max_calls,
            "calls": list(self.calls)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCallMemory':
        """反序列化"""
        memory = cls(max_calls=data.get("max_calls", 20))
        memory.calls.extend(data.get("calls", []))
        return memory


class SearchR1Memory:
    """Search-R1 完整记忆系统"""
    
    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 2048,
        max_tool_calls: int = 20
    ):
        """
        完整的记忆系统，包含对话和工具调用
        
        Args:
            max_turns: 最大对话轮数
            max_tokens: 最大上下文 token 数
            max_tool_calls: 最大工具调用数
        """
        self.conversation = ConversationMemory(max_turns, max_tokens)
        self.tool_calls = ToolCallMemory(max_tool_calls)
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.conversation.add_message("user", content)
    
    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict]] = None):
        """
        添加助手消息
        
        Args:
            content: 助手回复内容
            tool_calls: 工具调用列表（如果有）
        """
        self.conversation.add_message("assistant", content)
        
        # 记录工具调用
        if tool_calls:
            for call in tool_calls:
                self.tool_calls.add_call(
                    tool_name=call.get("name", ""),
                    arguments=call.get("arguments", {}),
                    result=call.get("result", "")
                )
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """获取完整的上下文消息（用于模型输入）"""
        return self.conversation.get_messages()
    
    def get_summary(self) -> str:
        """获取记忆摘要（用于奖励函数）"""
        messages = self.conversation.get_messages()
        recent_calls = self.tool_calls.get_recent_calls(n=3)
        
        summary = f"Conversation history: {len(messages)} messages, "
        summary += f"{self.conversation.turn_count} turns\n"
        summary += f"Recent tool calls: {len(recent_calls)}\n"
        
        return summary
    
    def should_search(self, query_hint: str) -> bool:
        """
        判断是否应该执行搜索（避免重复搜索）
        
        Args:
            query_hint: 可能的搜索查询
            
        Returns:
            是否应该搜索
        """
        return not self.tool_calls.has_recent_search(query_hint)
    
    def compress(self):
        """压缩记忆"""
        self.conversation.compress()
    
    def clear(self):
        """清空所有记忆"""
        self.conversation.clear()
        self.tool_calls.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "conversation": self.conversation.to_dict(),
            "tool_calls": self.tool_calls.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchR1Memory':
        """反序列化"""
        memory = cls()
        memory.conversation = ConversationMemory.from_dict(data.get("conversation", {}))
        memory.tool_calls = ToolCallMemory.from_dict(data.get("tool_calls", {}))
        return memory
    
    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SearchR1Memory':
        """从文件加载"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    # 测试
    print("Testing SearchR1Memory...")
    
    memory = SearchR1Memory(max_turns=5)
    
    # 模拟对话
    memory.add_user_message("What's the weather in Beijing?")
    memory.add_assistant_message(
        "Let me search for that.",
        tool_calls=[{
            "name": "web_search",
            "arguments": {"query": "Beijing weather today"},
            "result": "Sunny, 25°C"
        }]
    )
    memory.add_assistant_message("The weather in Beijing is sunny with 25°C.")
    
    memory.add_user_message("How about Shanghai?")
    
    # 检查记忆
    print("\nConversation messages:")
    for msg in memory.get_context_messages():
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    print(f"\nSummary:\n{memory.get_summary()}")
    
    print(f"\nShould search 'Beijing weather'? {memory.should_search('Beijing weather')}")
    print(f"Should search 'Shanghai weather'? {memory.should_search('Shanghai weather')}")
    
    # 测试序列化
    print("\nTesting serialization...")
    data = memory.to_dict()
    memory2 = SearchR1Memory.from_dict(data)
    assert len(memory2.get_context_messages()) == len(memory.get_context_messages())
    
    print("\n[OK] All tests passed!")

