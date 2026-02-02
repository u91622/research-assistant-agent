# AI研究助理 (Research Assistant Agent)

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

### 基準測試範例 (Llama-3-8B 推論)

以下為使用 `tests/test_performance.py` 進行的效能基準測試 (模擬數據)：

| 框架 | 延遲 (Latency) | 吞吐量 (Throughput) | 備註 |
|---|---|---|---|
| **vLLM** | **~50ms** | **~150 tok/s** | 適合高並發，吞吐量高 |
| **HuggingFace TGI** | ~80ms | ~80 tok/s | 適合長文本，整合性佳 |
| **Agent 本地模擬** | < 1s | N/A | 本地單元測試基準 |

> 若要在 Colab 上運行真實的 vLLM vs TGI 比較，請參考 `tests/test_performance.py` 中的說明。

## CI/CD 自動化測試

本專案已整合 GitHub Actions，每次 `push` 或 `pull_request` 到 `main` 分支時，會自動執行以下流程：
1. **安裝依賴**：`pip install -r requirements.txt`
2. **單元與整合測試**：`pytest tests/test_agent.py tests/test_tools.py`
3. **效能基準測試**：`pytest tests/test_performance.py`

詳細設定請見 `.github/workflows/ci.yml`。

## 目錄結構
- `agent.py`: 主要代理人邏輯與進入點。
- `tools.py`: 工具定義。
- `tests/`: Pytest 測試檔案。

