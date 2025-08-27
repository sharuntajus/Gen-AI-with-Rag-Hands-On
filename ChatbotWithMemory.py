# LangChain Core
from langchain_core.messages import HumanMessage, AIMessage

# Ollama LLM (local model)
from langchain_ollama import OllamaLLM

# Conversation memory modules
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Setup: Local Ollama LLM (adjust model name as needed)
llm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    max_tokens=200,
    top_p=0.9
)

# Initialize conversation memory (stores full chat history)
memory = ConversationBufferMemory(return_messages=True)

# Create the conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Sample interactive loop
test_inputs = [
    "Hello, I'm Sharun Tajus.",
    "I enjoy building AI applications.",
    "What projects would you recommend?",
    "What was my name again?",
    "Can you summarize our conversation?"
]

# Run simulation
print("\n=== Local Chat Simulation with Memory ===")
for i, input_text in enumerate(test_inputs):
    print(f"\nUser: {input_text}")
    response = conversation.invoke(input=input_text)
    print(f"AI: {response['response']}")

# Check the memory buffer
print("\n=== Memory Buffer ===")
print(conversation.memory.buffer)
