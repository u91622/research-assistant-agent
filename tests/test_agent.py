import pytest
import time
from unittest.mock import MagicMock, patch
import sys
import os
# 將父目錄加入 sys.path 以便匯入 agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import app
from langchain_core.messages import HumanMessage, AIMessage

# 模擬 LLM 以避免需要 API 金鑰並確保確定性測試
@pytest.fixture
def mock_llm_response():
    with patch("agent.ChatOpenAI") as mock_chat:
        mock_instance = mock_chat.return_value
        # 配置模擬以返回特定響應
        # 當被調用時，它應該返回一個 AIMessage
        mock_instance.bind_tools.return_value = mock_instance
        yield mock_instance

def test_math_workflow_success(mock_llm_response):
    """
    測試代理人是否能夠成功處理數學查詢。
    我們模擬 LLM 的決策來模擬流程。
    """
    # 模擬 LLM 決定調用工具
    mock_llm_response.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 2, "b": 3}, "id": "call_1"}]),
        AIMessage(content="The result is 6.")
    ]

    inputs = {"messages": [HumanMessage(content="What is 2 * 3?")]}
    config = {"configurable": {"thread_id": "test_1"}}
    
    # 執行圖表
    events = list(app.stream(inputs, config=config, stream_mode="values"))
    
    # 檢查我們是否得到最終回應
    final_message = events[-1]["messages"][-1]
    assert "6" in final_message.content
    assert len(events) > 1 # 應該有步驟

    assert len(events) > 1 # 應該有步驟

def test_add_workflow_success(mock_llm_response):
    """
    測試代理人是否能夠成功處理加法查詢。
    """
    # 模擬 LLM 決定調用工具
    mock_llm_response.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 10, "b": 20}, "id": "call_add_1"}]),
        AIMessage(content="The result is 30.")
    ]

    inputs = {"messages": [HumanMessage(content="What is 10 + 20?")]}
    config = {"configurable": {"thread_id": "test_add_1"}}
    
    # 執行圖表
    events = list(app.stream(inputs, config=config, stream_mode="values"))
    
    # 檢查我們是否得到最終回應
    final_message = events[-1]["messages"][-1]
    assert "30" in final_message.content

def test_search_workflow_latency(mock_llm_response):
    """
    基準測試代理人工作流程的延遲。
    """
    # 模擬 LLM 行為
    mock_llm_response.invoke.side_effect = [
        AIMessage(content="Searching...", tool_calls=[{"name": "search_duckduckgo", "args": {"query": "LangGraph"}, "id": "call_2"}]),
        AIMessage(content="LangGraph is a library...")
    ]

    # 有效地模擬工具輸出以避免部分搜尋調用（如果不需要或允許的話）
    # 我們要允許真實工具運行還是模擬它？
    # 對於延遲測試，我們可能希望模擬網絡調用以保持穩定，
    # 或者我們想測量真實延遲。
    # 鑑於“成功率和延遲的基本 pytest 測試”，通常意味著測量系統。
    # 但如果沒有金鑰/網絡，它可能會不穩定。我將在這個“基本”測試中模擬工具執行以保持穩定性。
    
    # 模擬 tools.DDGS (因為我們現在直接使用它)
    with patch("tools.DDGS") as mock_ddgs_cls:
        mock_ddgs_instance = mock_ddgs_cls.return_value
        mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
        # 設定 text 方法返回模擬資料
        mock_ddgs_instance.text.return_value = [
            {"title": "LangGraph", "href": "http://example.com", "body": "LangGraph result"}
        ]
        
        start_time = time.time()
        
        inputs = {"messages": [HumanMessage(content="Search for LangGraph")]}
        config = {"configurable": {"thread_id": "test_2"}}
        
        list(app.stream(inputs, config=config, stream_mode="values"))
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n工作流程延遲: {duration:.4f}s")
        # 斷言它相當快（模擬應該非常快）
        assert duration < 5.0 
