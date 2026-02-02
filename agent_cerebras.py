from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from tools import tools
import os

class AgentState(TypedDict):
    messages: list[BaseMessage]

def agent(state: AgentState):
    messages = state['messages']
    
    # 加入系統指令
    system_prompt = SystemMessage(content="""你是一位研究助理。你只能使用以下提供的工具：
1. multiply: 相乘兩個整數
2. add: 相加兩個整數
3. search_duckduckgo: 搜尋網路

請嚴格依照工具定義進行調用。""")
    
    input_messages = [system_prompt] + messages

    # 改用 ChatOpenAI 客戶端連接 Cerebras (OpenAI 兼容接口)
    # 這能解決 422 Tool call not found 的序列化問題
    # 使用 Cerebras 的 Llama-3.3-70b (目前最新的 70B 模型)
    # 此模型 ID 是 "llama-3.3-70b"，請勿使用舊版 ID
    model = ChatOpenAI(
        model="llama-3.3-70b", 
        temperature=0,
        api_key=os.environ.get("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1"
    )
    model = model.bind_tools(tools)
    response = model.invoke(input_messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")

def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print("Cerebras 版研究助理已啟動 (極速推論)。輸入 'quit' 退出。")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in app.stream(inputs, config=config, stream_mode="values"):
            event["messages"][-1].pretty_print()
