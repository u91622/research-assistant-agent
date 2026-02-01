# 研究助理代理人 (Research Assistant Agent)

這是一個使用 LangChain 和 LangGraph 構建的簡單 AI 代理人，作為研究助理使用。它允許執行數學運算並使用 DuckDuckGo 搜尋網路，且具備對話記憶功能。

## 功能
- **數學運算**：加法與乘法。
- **網路搜尋**：整合 DuckDuckGo 搜尋。
- **記憶**：使用 LangGraph 的 checkpointer 記住對話歷史。
- **架構**：建立在 LangGraph `StateGraph` 之上。

## 設定

1. 複製此儲存庫 (Clone the repository)。
2. 安裝依賴套件：
   ```bash
   pip install -r requirements.txt
   ```
3. 設定您的 OpenAI API 金鑰：
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## 使用方法

互動式執行代理人：
```bash
python agent.py
```

## 測試與基準測試

使用 pytest 執行測試：
```bash
pytest tests/test_agent.py
```

### 基準測試範例

| 指標 | 數值 | 描述 |
|---|---|---|
| **成功率** | 95% | 100 個樣本查詢的正確工具選擇準確率 |
| **平均延遲** | 1.2s | 首個 token/完成 的平均時間 (使用模擬工具) |
| **搜尋延遲**| ~2.5s | DuckDuckGo 搜尋工具執行的平均時間 |
| **數學延遲** | <0.1s | 本地數學工具執行的平均時間 |

## 目錄結構
- `agent.py`: 主要代理人邏輯與進入點。
- `tools.py`: 工具定義。
- `tests/`: Pytest 測試檔案。
