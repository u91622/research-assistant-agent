from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search import DDGS

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
        results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])

tools = [multiply, add, search_duckduckgo]
