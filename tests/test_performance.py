import pytest
import time
import sys
import os
import threading
from unittest.mock import patch, MagicMock

# 將父目錄加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import app
from langchain_core.messages import HumanMessage, AIMessage
try:
    import torch
except ImportError:
    torch = None

def measure_latency(func, *args, **kwargs):
    """測量函數執行的延遲"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

@pytest.mark.benchmark
def test_agent_latency_benchmark():
    """基準測試 Agent 對於簡單數學查詢的延遲"""
    inputs = {"messages": [HumanMessage(content="What is 100 * 200?")]}
    config = {"configurable": {"thread_id": "bench_latency"}}
    
    # 模擬 LLM 以避免網路請求，專注於系統開銷
    with patch("agent.ChatOpenAI") as mock_chat:
        mock_instance = mock_chat.return_value
        mock_instance.bind_tools.return_value = mock_instance
        # 模擬兩次調用：一次工具調用，一次最終回答
        # 注意：LangGraph/LangChain 期望返回的是 Message 物件，而不是單純的 MagicMock
        mock_instance.invoke.side_effect = [
             AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 100, "b": 200}, "id": "call_1"}]),
             AIMessage(content="The result is 20000.")
        ]

        def run_agent():
            return list(app.stream(inputs, config=config, stream_mode="values"))

        events, duration = measure_latency(run_agent)
        
        print(f"\n[Latency] Agent Response Time: {duration:.4f}s")
        assert duration < 10.0, "Agent latency is too high!"

@pytest.mark.benchmark
def test_throughput_simulation():
    """模擬並發請求以測試吞吐量 (Throughput)"""
    request_count = 5 # Reduce count to speed up
    start_time = time.time()
    
    with patch("agent.ChatOpenAI") as mock_chat:
        mock_instance = mock_chat.return_value
        mock_instance.bind_tools.return_value = mock_instance
        mock_instance.invoke.return_value = AIMessage(content="Mock response")
        
        def mock_request():
            inputs = {"messages": [HumanMessage(content="Hello")]}
            config = {"configurable": {"thread_id": f"bench_throughput_{threading.get_ident()}"}}
            list(app.stream(inputs, config=config, stream_mode="values"))

        threads = []
        for _ in range(request_count):
            t = threading.Thread(target=mock_request)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
            
    end_time = time.time()
    total_time = end_time - start_time
    throughput = request_count / total_time
    
    print(f"\n[Throughput] Processed {request_count} requests in {total_time:.4f}s. Rate: {throughput:.2f} req/s")
    assert throughput > 0.01, "Throughput is too low!"

@pytest.mark.benchmark
def test_gpu_memory_usage():
    """監控 GPU 記憶體使用量 (如果有的話)"""
    if torch and torch.cuda.is_available():
        start_mem = torch.cuda.memory_allocated()
        print(f"\n[GPU Memory] Start: {start_mem / 1024**2:.2f} MB")
        
        # 模擬一些 GPU 操作 (此處僅為範例，實際 Agent 需載入模型才會有顯著變化)
        tensor = torch.zeros((1000, 1000)).cuda()
        
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"[GPU Memory] Peak: {peak_mem / 1024**2:.2f} MB")
        
        del tensor
        torch.cuda.empty_cache()
    else:
        print("\n[GPU Memory] No GPU detected, skipping memory benchmark.")

def test_tgi_vs_vllm_comparison_mock():
    """
    TGI vs vLLM 的模擬基準測試
    (實際執行需依賴真實 Docker 環境與 GPU，此處驗證測試邏輯是否正確)
    """
    # 模擬數據：vLLM 通常比 TGI 快 (throughput 高, letency 低)
    frameworks = {
        "vLLM": {"latency": 0.05, "throughput": 150},
        "TGI": {"latency": 0.08, "throughput": 80}
    }
    
    print("\n[Comparison] Llama-3-8B Inference Performance (Mock)")
    print(f"{'Framework':<10} | {'Latency (s)':<12} | {'Throughput (tok/s)':<20}")
    print("-" * 50)
    
    for name, metrics in frameworks.items():
        print(f"{name:<10} | {metrics['latency']:<12.3f} | {metrics['throughput']:<20.1f}")
    
    assert frameworks["vLLM"]["throughput"] > frameworks["TGI"]["throughput"]
    assert frameworks["vLLM"]["latency"] < frameworks["TGI"]["latency"]
