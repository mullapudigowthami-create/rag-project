from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor
from langchain.tools import tool

# Connect to Claude AI
llm = ChatAnthropic(api_key="your-api-key")

# Create a simple tool
@tool
def search_web(query: str):
    """Search the web for information"""
    return f"Results for {query}"

# Build agent
tools = [search_web]
agent = AgentExecutor(agent=llm, tools=tools)

# Run agent
response = agent.invoke({"input": "What is Swachh Bharat?"})
print(response)