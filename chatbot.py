import random

def get_response(user_input):
    user_input = user_input.lower()
    
    # Define rules and keywords
    if "hello" in user_input or "hi" in user_input:
        return random.choice(["Hi there!", "Hello!", "Hey! How can I help?"])
    elif "how are you" in user_input:
        return "I'm just a bunch of code, but I'm doing great!"
    elif "weather" in user_input:
        return "I don't have a window, but it looks like a sunny day for coding!"
    elif "bye" in user_input:
        return "Goodbye! Have a great day."
    else:
        return "I'm not sure I understand. Can you try asking about the weather?"

# Main conversation loop
print("Chatbot: Hi! Type 'bye' to exit.")
while True:
    user_text = input("You: ")
    if user_text.lower() == 'bye':
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_text)
    print(f"Chatbot: {response}")
