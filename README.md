# AI研究助理 (Research Assistant Agent)

這是一個使用 **LangChain** 和 **LangGraph** 構建的 AI 研究助理代理人。
專案核心採用 **Cerebras** (全球最快推論引擎) 作為主要驅動力，並提供整合式 Web 介面。

## ✨ 主要功能
- **極速推論**：整合 **Cerebras Llama-3.3-70B** 與 **GP-OSS-120B** 模型。
- **數學運算**：精確執行加法與乘法工具。
- **網路搜尋**：整合 DuckDuckGo 搜尋即時資訊。
- **對話記憶**：具備完整的對話上下文記憶功能。
- **動態切換**：Web UI 支援即時切換不同模型。

## ⚙️ 設定 Setup

1. **複製儲存庫 (Clone the repository)**
2. **安裝依賴套件**：
   ```bash
   pip install -r requirements.txt
   ```
3. **(核心) 設定 Cerebras API 金鑰**：
   - 您可以直接在 Web UI 介面輸入。
   - 或者在終端機預先設定環境變數：
     ```bash
     export CEREBRAS_API_KEY="csk-..."
     ```

## 🚀 使用方法 Usage

### 1. 啟動整合版 Web UI (推薦)
這是最完整的使用方式，提供視覺化介面與模型選單。

```bash
streamlit run app.py
```

> **🌟 特色功能**：
> 左側側邊欄提供 **「模型選擇選單」**，您可以自由切換：
> - **Cerebras (Llama-3.3-70B)**：最新旗艦模型，兼具速度與穩定性。
> - **Cerebras (GPT-OSS-120B)**：強大的開源模型。

### 2. 互動式 CLI 模式
若您想在終端機直接測試 (預設使用 Llama-3.3-70B)：
```bash
python agent.py
```

## 📊 效能評測 (Benchmarks)

本專案包含關於 **Llama-3-8B** 等模型的深入效能測試。

- **測試檔案**: `benchmark_colab.ipynb`
- **測試內容**: 詳細記錄了在 Google Colab (T4 GPU) 環境下，使用 **vLLM** 與 **Hugging Face TGI** 的推論吞吐量比較。
- **注意**：由於本地端 (Local) 硬體通常無法運行 70B 參數等級的模型，因此我們使用 Colab 進行伺服器端的極限壓力測試。請直接查看 Notebook 內的圖表數據。

## CI/CD 與測試
本專案整合 GitHub Actions 自動化測試：
```bash
# 執行單元測試
pytest tests/test_agent.py
```

## 目錄結構
- `app.py`: Streamlit 網頁應用程式 (單一整合入口)。
- `agent.py`: LangGraph 代理人核心邏輯。
- `tools.py`: 自定義工具 (Search, Math)。
- `tests/`: 測試腳本。

