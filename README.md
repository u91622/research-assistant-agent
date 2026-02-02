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

互动式執行代理人 (CLI)：
```bash
python agent.py
```

啟動網頁介面 (Web UI)：
```bash
# 啟動整合版介面 (Cerebras Llama / GPT-OSS)
streamlit run app.py
```

## 測試與基準測試

使用 pytest 執行測試：
```bash
pytest tests/test_agent.py
```

### 實際效能基準測試 (Llama-3.2-1B)

本專案使用 Google Colab (T4 GPU) 對推論引擎進行了實測，數據如下：

| 測試環境 | 推論框架 | 模型 | 吞吐量 (Throughput) |
|---|---|---|---|
| **Google Colab (T4)** | **vLLM** | **Llama-3.2-1B** | **~1950 tokens/s** |
| 本地測試 (WSL) | 模擬 (Mock) | N/A | < 1s (系統延遲) |

> 這些數據證明了使用 vLLM 進行高度併發推論的效能優勢。詳細測試腳本請見 `benchmark_colab.ipynb`。

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

