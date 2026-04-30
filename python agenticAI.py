from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub  
from langchain.tools import tool

# Connect to Gemini AI
llm = ChatGoogleGenerativeAI( model="gemini-1.5-flash",api_key="AIzaSyCmepurVUp3c3VWzKCIJwZ7ZPucUjdiOB0")
# Create a simple tool
@tool
def search_web(query: str):
    """Search the web for information"""
    return f"Results for {query}"

# Build agent
tools = [search_web]

# Get Standard Prompt for ReAct agents
prompt= hub.pull("hwchase17/react")

# Construct agent logic
agent= create_react_agent(llm, tools, prompt)

# Create the executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
response = agent_executor.invoke({"input": "What is Swachh Bharat?"})
print(response["output"])