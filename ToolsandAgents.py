# from langchain_core.tools import Tool
# from langchain.tools import tool  # For decorator-style tool creation
# from langchain_experimental.utilities import PythonREPL
#
# python_repl = PythonREPL()
#
# python_calculator = Tool(
#     name="Python Calculator",
#     func=python_repl.run,
#     description="Useful for calculations or executing Python code. Input should be valid Python code."
# )
#
# # Example usage:
# python_calculator.invoke("a = 3; b = 1; print(a + b)")  # Output: 4
#
# @tool
# def search_weather(location: str):
#     """Search for the current weather in the specified location."""
#     # Stubbed response – normally you'd call a real API here
#     return f"The weather in {location} is currently sunny and 72°F."
#
# tools = [python_calculator, search_weather]
#
# from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
#
# llm = OllamaLLM(model="mistral")
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )
#
# agent.invoke("What is 5 * (3 + 2)? Also, what's the weather in Paris?")

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# Create a simple calculator tool
def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Create a text formatting tool
def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: [format_type]: [text]
    where format_type is uppercase, lowercase, or titlecase.

    Examples:
    - uppercase: hello world -> HELLO WORLD
    - lowercase: HELLO WORLD -> hello world
    - titlecase: hello world -> Hello World
    """
    try:
        # Handle the case where the entire string is passed
        if ":" in text:
            format_type, content = text.split(":", 1)
            format_type = format_type.strip().lower()
            content = content.strip()
        else:
            # If no colon, assume they want titlecase
            return f"Missing format. Example: titlecase: {text} -> {text.title()}"

        if format_type == "uppercase":
            return content.upper()
        elif format_type == "lowercase":
            return content.lower()
        elif format_type == "titlecase":
            return content.title()
        else:
            return f"Unknown format {format_type}. Use: uppercase, lowercase, or titlecase"

    except Exception as e:
        return f"Error formatting text: {str(e)}"


# Create Tool objects for our functions
tools = [
    Tool(
        name="calculator",
        func=calculator,
        description="Useful for performing simple math calculations"
    ),
    Tool(
        name="format_text",
        func=format_text,
        description="Useful for formatting text to uppercase, lowercase, or titlecase"
    )
]

# Create a simple prompt template
# Note the added {tool_names} variable which was missing before
prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.
You have access to these tools:

{tools}

The available tools are: {tool_names}

Follow this format:

Question: the user's question
Thought: think about what to do
Action: the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result from the tool
Thought: I now know the final answer
Final Answer: your final answer to the user's question

Question: {input}
{agent_scratchpad}
"""
llama_llm=OllamaLLM(model="mistral")
# Create the agent and executor
prompt = PromptTemplate.from_template(prompt_template)
agent = create_react_agent(
    llm=llama_llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Test with simple questions
test_questions = [
    "What is 25 + 63?",  # The agent will be able to answer this question
    "Can you convert 'hello world' to uppercase?",  # The agent might be able to answer this question
    # However, it is not guaranteed due to incorrect input format
    "Calculate 15 * 7",  # The agent will be able to answer this question
    "titlecase: langchain is awesome",  # The agent will be able to answer this question
]

# Run the tests
for question in test_questions:
    print(f"\n===== Testing: {question} =====")
    result = agent_executor.invoke({"input": question})
    print(f"Final Answer: {result['output']}")