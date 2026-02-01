import pytest
import sys
import os
from unittest.mock import patch

# 將父目錄加入 sys.path 以便匯入 tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import multiply, add, search_duckduckgo

def test_multiply():
    """測試乘法工具"""
    assert multiply.invoke({"a": 2, "b": 3}) == 6
    assert multiply.invoke({"a": -1, "b": 5}) == -5
    assert multiply.invoke({"a": 0, "b": 10}) == 0

def test_add():
    """測試加法工具"""
    assert add.invoke({"a": 2, "b": 3}) == 5
    assert add.invoke({"a": -1, "b": 1}) == 0
    assert add.invoke({"a": 10, "b": 20}) == 30

def test_search_duckduckgo_success():
    """測試 DuckDuckGo 搜尋成功的情況"""
    with patch("tools.DDGS") as mock_ddgs_cls:
        mock_ddgs_instance = mock_ddgs_cls.return_value
        mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
        
        # 模擬返回資料
        mock_ddgs_instance.text.return_value = [
            {"title": "Test Title", "href": "http://test.com", "body": "Test Body"}
        ]
        
        result = search_duckduckgo.invoke({"query": "test"})
        assert "Title: Test Title" in result
        assert "Link: http://test.com" in result
        assert "Snippet: Test Body" in result

def test_search_duckduckgo_no_results():
    """測試 DuckDuckGo 搜尋無結果的情況"""
    with patch("tools.DDGS") as mock_ddgs_cls:
        mock_ddgs_instance = mock_ddgs_cls.return_value
        mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
        
        # 模擬返回空列表
        mock_ddgs_instance.text.return_value = []
        
        result = search_duckduckgo.invoke({"query": "nothing"})
        assert result == "No results found."
