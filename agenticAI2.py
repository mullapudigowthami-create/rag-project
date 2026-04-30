from crewai import Agent, Task, Crew, LLM

# Configure Ollama LLM
llm=LLM(model="ollama/llama3.2",base_url="http://localhost:11434")

# Define Agents
researcher = Agent(
    role="Researcher",
    goal="Research and find information about any topic",
    backstory="You are an expert researcher with deep knowledge in many fields",
    llm=llm,
    verbose=True
)

calculator = Agent(
    role="Calculator",
    goal="Solve mathematical problems accurately",
    backstory="You are a mathematics expert who solves calculations precisely",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Summarize and present information clearly",
    backstory="You are an expert writer who presents information in simple language",
    llm=llm,
    verbose=True
)

# Define Tasks
research_task = Task(
    description="Research about Swachh Bharat Mission and summarize it",
    expected_output="A clear summary of Swachh Bharat Mission",
    agent=researcher
)

calculate_task = Task(
    description="Calculate 10+20 and explain the result",
    expected_output="The answer to 10+20 with explanation",
    agent=calculator
)

write_task = Task(
    description="Combine the research and calculation results into a nice report",
    expected_output="A well written report combining both results",
    agent=writer
)

# Create Crew
crew = Crew(
    agents=[researcher, calculator, writer],
    tasks=[research_task, calculate_task, write_task],
    verbose=True
)

# Run the Crew
print("AI Agent Crew Started!")
result = crew.kickoff()
print("\nFinal Result:")
print(result)