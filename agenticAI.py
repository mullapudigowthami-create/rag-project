from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.tools import tool

# Connect to Claude AI
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(api_key="AIzaSyCmepurVUp3c3VWzKCIJwZ7ZPucUjdiOB0", model="gemini-pro")
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