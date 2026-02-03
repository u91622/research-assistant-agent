import streamlit as st
import os
import sys
import uuid

# 將當前目錄加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_engine import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

# 設定頁面資訊
st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="centered")
st.title("AI Research Assistant (v2.1)")
st.caption("🚀 支援 Math, Search, 以及 **Native AutoML** (Scikit-Learn) - Reloaded")

# 側邊欄：模型選擇與設定
with st.sidebar:
    st.header("設定")
    
    # 1. 模型選單
    model_option = st.selectbox(
        "選擇模型 / Select Model",
        (
            "Cerebras (Llama-3.3-70B)",
            "Cerebras (GPT-OSS-120B)"
        )
    )
    
    # 側邊欄按鈕
    if st.button("🗑️ 清除對話 (Reset)", help="若遇到 422 錯誤或卡住，請點此重置"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # 2. 設定 Cerebras
    provider = "cerebras"
    if "Llama" in model_option:
        model_name = "llama-3.3-70b"
    else:
        model_name = "gpt-oss-120b"
        
    api_key = st.text_input("Cerebras API Key", type="password")
    if api_key:
        os.environ["CEREBRAS_API_KEY"] = api_key

    # 3. 檔案上傳區
    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        # 確保 data 目錄存在
        if not os.path.exists("data"):
            os.makedirs("data")
            
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: data/{uploaded_file.name}")
        st.caption("Tell agent: 'Train on uploaded file'")

# 初始化 Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 thread_id (每次重新整理或切換模型時可能需要注意 ID，但這裡我們先保持持久化)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 顯示對話歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理使用者輸入
if prompt := st.chat_input("Input message..."):
    # 1. 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 檢查 Key 是否存在
            if not os.environ.get("CEREBRAS_API_KEY"):
                st.warning("請輸入 Cerebras API Key")
                st.stop()
                
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            # 設定 Config (傳遞模型參數)
            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id,
                    "provider": provider,
                    "model_name": model_name
                }
            }
            
            # 串流回應
            # 不顯示 Spinner 文字，僅顯示轉圈圈 (預設行為) 或自訂空 spinner
            with st.spinner():
                for event in agent_app.stream(inputs, config=config, stream_mode="values"):
                    if "messages" in event:
                        latest_msg = event["messages"][-1]
                        if isinstance(latest_msg, AIMessage) and latest_msg.content:
                            full_response = latest_msg.content
                            message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
