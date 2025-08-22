from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence

#Set up Mistral model from Ollama
llm = OllamaLLM(model="mistral")

# Define prompt
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question in detail clearly and concisely :\n\nQuestion: {question}"
)
# Create RunnableSequence (no more LLMChain)
chain = prompt | llm

# Use invoke (instead of .run)
response = chain.invoke({"question": "By what you have built on mistral?"})


promptwithContext = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant. Use the provided context to answer the user's question in detail.

Context:
{context}

Question:
{question}

Answer in a clear, friendly, and professional tone. If the context is not helpful, use general knowledge.
"""
)
# Create RunnableSequence (no more LLMChain)
chain = prompt | llm
chainwithContext = promptwithContext | llm
inputs = {
    "context": "User learning GenAI, he has 3 years of experience in DotNet full stack development.",
    "question": "How to speed up your response time?"
}

# Use invoke (instead of .run)
response = chain.invoke({"question": "By what you have built on mistral?"})
responseForContext = chainwithContext.invoke(inputs)

print("\nResponse:\n", response)
print("\nResponse with Context:\n", responseForContext)
