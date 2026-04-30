from google import genai

client = genai.Client(api_key="AIzaSyCmepurVUp3c3VWzKCIJwZ7ZPucUjdiOB0")

def search_web(query):
    return f"Search results for: {query}"

def calculate(expression):
    return str(eval(expression))

def get_weather(city):
    return f"Weather in {city}: Sunny, 30C"

def run_agent(user_input):
    prompt = f" You are AI agent.Answer this: {user_input}"

    for attempt in range(3):
      try:
        response = client.models.generate_content(
          model ='gemini-2.0-flash',
          contents=prompt
    )
    return response.text
except Exception as e:
    if "RESOURCE_EXHAUSTED"in str(e):
        print(f"Rate limited, waiting 60 seconds....")
        time.sleep(60)
      else:
    raise e
return "Failed after retries"   

print("AI Agent Started!")
result=run_agent("What is Swachh Bharat and What is 10+20?")
print(result)