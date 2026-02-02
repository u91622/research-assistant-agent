from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from tools import tools # 從我們的 tools.py 匯入工具

# 定義狀態
class AgentState(TypedDict):
    messages: list[BaseMessage]

from langchain_core.runnables import RunnableConfig
import os


# 定義代理人
def agent(state: AgentState, config: RunnableConfig):
    """
    主要的代理人節點，使用工具調用 LLM。
    支援透過 config 切換不同模型提供者。
    """
    messages = state['messages']
    
    # 讀取配置
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", "openai")
    model_name = configurable.get("model_name", "gpt-3.5-turbo")
    
    # 根據 provider 初始化模型
    # 目前僅支援 Cerebras (使用 OpenAI 兼容協議)
    if provider == "cerebras":
        model = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
    else:
        # Fallback to Cerebras just in case or raise error, but for now default to Cerebras Llama
        model = ChatOpenAI(
            model="llama-3.3-70b",
            temperature=0,
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

# 定義工具節點
tool_node = ToolNode(tools)

# 定義圖表
workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

# 新增邊
workflow.add_edge(START, "agent")

def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # 如果 LLM 返回 tool_calls，我們進入 tools
    if last_message.tool_calls:
        return "tools"
    # 否則我們結束
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 初始化記憶
memory = MemorySaver()

# 編譯圖表
app = workflow.compile(checkpointer=memory)

# 如果直接執行，用於演示的簡單進入點
if __name__ == "__main__":
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("研究助理代理人已啟動。輸入 'quit' 退出。")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in app.stream(inputs, config=config, stream_mode="values"):
            # 串流輸出
            message = event["messages"][-1]
            if isinstance(message, tuple):
                 print(message)
            else:
                 message.pretty_print()
