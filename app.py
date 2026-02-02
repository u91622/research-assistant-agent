import streamlit as st
import os
import sys

# 將當前目錄加入 sys.path 以便匯入 agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

# 設定頁面資訊
st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="centered")

st.title("🤖 AI Research Assistant")
st.markdown("我可以幫您進行 **數學運算** 與 **網路搜尋**！")

# 側邊欄：API Key 設定
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("OpenAI API Key", type="password", help="如果您沒有在環境變數設定，請在此輸入")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("### 關於")
    st.markdown("此專案展示了 LangGraph Agent 的能力，包含：")
    st.markdown("- 工具調用 (Math, Search)")
    st.markdown("- 對話記憶 (Memory)")
    st.markdown("- 串流回應 (Streaming)")

# 初始化 Streamlit session state 來儲存對話歷史 (僅用於 UI 顯示)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示這一次 Session 的對話紀錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理使用者輸入
if prompt := st.chat_input("請問... (例如：查一下 LangGraph 是什麼？)"):
    # 1. 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 準備輸入：只傳送最新的使用者訊息，歷史由 Agent 的 Memory 機制處理
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            # 設定 thread_id 以便 Agent 辨識這是同一個對話
            config = {"configurable": {"thread_id": "streamlit_user_session"}}
            
            # 使用 stream 來獲取回應
            # stream_mode="values" 會回傳每個步驟更新後的完整 state
            for event in agent_app.stream(inputs, config=config, stream_mode="values"):
                if "messages" in event:
                    latest_msg = event["messages"][-1]
                    # 只顯示 AI 的最終回應，或是工具調用的過程也可以考慮顯示 (這裡先顯示最終回應)
                    if isinstance(latest_msg, AIMessage) and latest_msg.content:
                        full_response = latest_msg.content
                        message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # 存入 UI 歷史
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
