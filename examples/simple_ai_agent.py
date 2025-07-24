from crewgraph_ai import AgentWrapper

# Define a simple agent class
class SimpleAIAgent(AgentWrapper):
    def __init__(self, name="simple_agent", goal="Respond to user queries"):
        super().__init__(name=name, goal=goal)

    def act(self, input_data):
        # Simple logic: echo the input with a greeting
        return f"Hello! You said: {input_data}"

# Example usage
if __name__ == "__main__":
    agent = SimpleAIAgent()
    user_input = input("Enter your message: ")
    response = agent.act(user_input)
    print(response)
