from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from tools import tools # 從我們的 tools.py 匯入工具

from langgraph.graph.message import add_messages

# 定義狀態
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_core.runnables import RunnableConfig
import os


# 定義代理人
def agent(state: AgentState, config: RunnableConfig):
    """
    主要的代理人節點，使用工具調用 LLM。
    支援透過 config 切換不同模型提供者。
    """
    messages = state['messages']
    
    # 定義系統提示
    system_prompt = SystemMessage(content="""You are a helpful AI Research Assistant.
You have access to the following tools:
1. multiply: Multiply two integers.
2. add: Add two integers.
3. search_duckduckgo: Search the web.
4. train_tabular_model: AutoML tool to train machine learning models on tabular data.

When asked to analyze data or train a model, use 'train_tabular_model'.
For Titanic dataset, use 'openml:40945'. For Iris, use 'openml:61'.
If the user mentions an uploaded file, look for it in the 'data/' directory (e.g., 'data/filename.csv').
Always use the tools provided. Do not halllucinate answers for math or data training.""")
    
    # 安全地建構輸入訊息列表 (不要修改原始 state)
    # 檢查第一則訊息是否已經是 SystemMessage
    if messages and isinstance(messages[0], SystemMessage):
        # 如果已經有了，我們用新的取代它 (建立新列表)
        input_messages = [system_prompt] + messages[1:]
    else:
        # 如果沒有，插入在最前面
        input_messages = [system_prompt] + messages
    
    # 讀取配置
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", "openai")
    model_name = configurable.get("model_name", "gpt-3.5-turbo")
    
    # ... (Model init code)
    
    # 這裡省略模型初始化程式碼，因為上面 context 沒包含到，但我們只需要確保下面 invoke 用的是 input_messages

    # (Re-stating the model init block to be safe in replacement if needed, 
    # but since I am using replace_file_content with context, I will just focus on the logic block)
    
    # 根據 provider 初始化模型
    if provider == "cerebras":
        model = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
    else:
        model = ChatOpenAI(
            model="llama-3.3-70b",
            temperature=0,
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )

    model = model.bind_tools(tools)
    response = model.invoke(input_messages) # 使用 input_messages
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
