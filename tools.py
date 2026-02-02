from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search import DDGS
import warnings

# 忽略 DuckDuckGoSearch 的更名警告與可能的資源警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=ResourceWarning)

@tool
def multiply(a: int, b: int) -> int:
    """相乘兩個整數。"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """相加兩個整數。"""
    return a + b

@tool
def search_duckduckgo(query: str) -> str:
    """使用 DuckDuckGo 搜尋網路。"""
    with DDGS() as ddgs:
        # 減少搜尋結果數量以降低 Token 消耗 (避免 Groq 免費版 Rate Limit)
        results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])

tools = [multiply, add, search_duckduckgo]
