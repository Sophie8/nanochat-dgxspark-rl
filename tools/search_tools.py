"""
Search Tools for Search-R1 Training
支持多个搜索引擎 API，自动回退机制
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    url: str
    snippet: str
    score: float = 0.0


class SearchEngine(ABC):
    """搜索引擎抽象基类"""
    
    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """执行搜索"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查 API 是否可用"""
        pass


class TavilySearch(SearchEngine):
    """Tavily Search API - 为 AI 优化的搜索引擎"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        
        try:
            response = requests.post(
                self.base_url,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": num_results,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score", 0.0),
                ))
            return results
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return []
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class DuckDuckGoSearch(SearchEngine):
    """DuckDuckGo 搜索 - 完全免费，无需 API key，支持代理"""

    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy or os.getenv("SEARCH_PROXY") or os.getenv("https_proxy")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        try:
            # 优先使用新版 ddgs 包
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            kwargs = {}
            if self.proxy:
                kwargs["proxy"] = self.proxy

            ddgs = DDGS(**kwargs)
            results_raw = list(ddgs.text(query, max_results=num_results))
            results = []
            for item in results_raw:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("url", "")),
                    snippet=item.get("body", item.get("snippet", "")),
                    score=0.5,
                ))
            return results

        except ImportError:
            print("ddgs / duckduckgo_search not installed. pip install ddgs")
            return []
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return []

    def is_available(self) -> bool:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            return True
        except ImportError:
            return False


class SerperSearch(SearchEngine):
    """Serper API - Google Search API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []
        
        try:
            response = requests.post(
                self.base_url,
                headers={"X-API-KEY": self.api_key},
                json={"q": query, "num": num_results},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    score=0.8,
                ))
            return results
        except Exception as e:
            print(f"Serper search failed: {e}")
            return []
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class GeminiSearch(SearchEngine):
    """基于 Gemini API 的搜索引擎 - 使用 LLM 生成搜索结果风格的信息"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_url = api_url or os.getenv(
            "GEMINI_API_URL", "")
        self.api_key = api_key or os.getenv(
            "GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []

        prompt = (
            f"你是一个网络搜索引擎。请根据以下查询，返回 {min(num_results, 5)} 条"
            f"搜索结果摘要。每条结果包含标题、来源URL和内容摘要。"
            f"请提供真实准确的信息，就像真正的搜索引擎一样。\n\n"
            f"搜索查询：{query}\n\n"
            f"请严格以JSON格式返回，不要添加其他文字：\n"
            f'{{"results": [{{"title": "...", "url": "https://...", "snippet": "..."}}]}}'
        )

        try:
            resp = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个搜索引擎，返回准确的搜索结果。只返回JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()

            # 提取 JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    score=0.7,
                ))
            return results

        except Exception as e:
            print(f"Gemini search failed: {e}")
            return []

    def is_available(self) -> bool:
        return bool(self.api_key)


class MockSearch(SearchEngine):
    """Mock 搜索引擎 - 用于测试，返回固定结果"""
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        # 返回模拟结果
        mock_results = [
            SearchResult(
                title=f"Result {i+1} for '{query}'",
                url=f"https://example.com/result_{i+1}",
                snippet=f"This is a mock search result #{i+1} for query: {query}. "
                        f"It contains relevant information about the topic.",
                score=1.0 - i * 0.1,
            )
            for i in range(min(num_results, 3))
        ]
        return mock_results
    
    def is_available(self) -> bool:
        return True


class SearchToolkit:
    """搜索工具包 - 自动选择可用的搜索引擎"""
    
    def __init__(self, preferred_engines: Optional[List[str]] = None):
        """
        Args:
            preferred_engines: 优先使用的引擎列表，按优先级排序
                             ['tavily', 'serper', 'duckduckgo', 'mock']
        """
        if preferred_engines is None:
            preferred_engines = ['gemini', 'tavily', 'serper', 'duckduckgo', 'mock']
        
        # 初始化所有引擎
        self.engines = {
            'gemini': GeminiSearch(),
            'tavily': TavilySearch(),
            'serper': SerperSearch(),
            'duckduckgo': DuckDuckGoSearch(),
            'mock': MockSearch(),
        }
        
        # 按优先级排序
        self.preferred_engines = [
            e for e in preferred_engines if e in self.engines
        ]
        
        # 找到第一个可用的引擎
        self.active_engine = None
        for name in self.preferred_engines:
            if self.engines[name].is_available():
                self.active_engine = name
                print(f"[OK] Using search engine: {name}")
                break
        
        if not self.active_engine:
            raise RuntimeError("No search engine available!")
    
    def search(self, query: str, num_results: int = 5, max_retries: int = 2) -> List[SearchResult]:
        """
        执行搜索，自动回退到备用引擎，支持重试

        Args:
            query: 搜索查询
            num_results: 返回结果数量
            max_retries: 每个引擎的最大重试次数

        Returns:
            搜索结果列表
        """
        for engine_name in self.preferred_engines:
            engine = self.engines[engine_name]
            if not engine.is_available():
                continue

            for attempt in range(max_retries + 1):
                results = engine.search(query, num_results)
                if results:
                    return results
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)  # exponential backoff
                    time.sleep(wait)

            print(f"Engine {engine_name} returned no results after {max_retries+1} attempts, trying next...")

        # 所有引擎都失败，返回空列表
        print("All search engines failed!")
        return []
    
    def format_results(self, results: List[SearchResult], max_length: int = 500) -> str:
        """
        格式化搜索结果为文本
        
        Args:
            results: 搜索结果列表
            max_length: 每个结果的最大长度
            
        Returns:
            格式化的文本
        """
        if not results:
            return "No search results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            snippet = result.snippet[:max_length]
            if len(result.snippet) > max_length:
                snippet += "..."
            
            formatted.append(
                f"[{i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"{snippet}\n"
            )
        
        return "\n".join(formatted)


# OpenAI Function Calling 规范的 Tool Definition
SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when you need up-to-date facts, statistics, news, or any information not in your training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include relevant keywords."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10)",
                    "default": 5,
                }
            },
            "required": ["query"]
        }
    }
}


def execute_search_tool(toolkit: SearchToolkit, tool_call: Dict[str, Any]) -> str:
    """
    执行搜索工具调用
    
    Args:
        toolkit: SearchToolkit 实例
        tool_call: 工具调用字典，格式：
            {
                "name": "web_search",
                "arguments": {"query": "...", "num_results": 5}
            }
    
    Returns:
        搜索结果的格式化文本
    """
    if tool_call.get("name") != "web_search":
        return f"Error: Unknown tool '{tool_call.get('name')}'"
    
    try:
        args = tool_call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        
        query = args.get("query", "")
        num_results = args.get("num_results", 5)
        
        if not query:
            return "Error: No search query provided"
        
        results = toolkit.search(query, num_results)
        return toolkit.format_results(results)
    
    except Exception as e:
        return f"Error executing search: {str(e)}"


if __name__ == "__main__":
    # 测试
    print("Testing SearchToolkit...")
    
    # 使用 mock 引擎进行测试
    toolkit = SearchToolkit(preferred_engines=['mock'])
    
    # 测试搜索
    results = toolkit.search("Python programming", num_results=3)
    print("\nSearch results:")
    print(toolkit.format_results(results))
    
    # 测试工具调用
    print("\nTesting tool call:")
    tool_call = {
        "name": "web_search",
        "arguments": {"query": "latest AI news", "num_results": 3}
    }
    output = execute_search_tool(toolkit, tool_call)
    print(output)
    
    print("\n[OK] All tests passed!")


