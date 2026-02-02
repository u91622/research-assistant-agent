import streamlit as st
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_cerebras import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AI Research Assistant (Cerebras)", page_icon="🚀", layout="centered")

st.title("🚀 AI Research Assistant (Cerebras Inference)")
st.markdown("體驗全球最快的 AI 推論引擎！使用 **Llama-3.3-70B** 模型 (最新旗艦)。")

with st.sidebar:
    st.header("設定")
    cerebras_api_key = st.text_input("Cerebras API Key", type="password", help="請至 https://cloud.cerebras.ai/ 取得 Key")

    if cerebras_api_key:
        os.environ["CEREBRAS_API_KEY"] = cerebras_api_key

if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("試試問我：解釋量子力學的核心概念？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if not os.environ.get("CEREBRAS_API_KEY"):
                st.warning("請在左側側邊欄輸入 Cerebras API Key 才能開始對話喔！")
                st.stop()
                
            inputs = {"messages": [HumanMessage(content=prompt)]}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.spinner("Cerebras 極速運算中..."):
                for event in agent_app.stream(inputs, config=config, stream_mode="values"):
                    if "messages" in event:
                        latest_msg = event["messages"][-1]
                        if isinstance(latest_msg, AIMessage) and latest_msg.content:
                            full_response = latest_msg.content
                            message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
