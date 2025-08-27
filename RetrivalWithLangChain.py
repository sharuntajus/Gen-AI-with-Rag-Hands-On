from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Load a document from the LangChain docs
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
documents = loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. Set up HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create a Chroma vector store from the chunks
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="langchain_web_docs",
    persist_directory="./chroma_store"  # optional: makes it persistent
)

# 5. Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. Configure Ollama LLM (e.g. Mistral)
llm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    max_tokens=300,
    top_p=0.9
)

# 7. Set up RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Test with a few queries
test_queries = [
    "What is LangChain?",
    "How do retrievers work?",
    "Why is document splitting important?"
]

for query in test_queries:
    print(f" Query: {query}")
    result = qa.invoke(query)
    print(f" Answer: {result['result']}")

    print("\n Sources:")
    for i, doc in enumerate(result['source_documents']):
        print(f" - [{i + 1}] {doc.metadata.get('source', 'Unknown')} | Preview: {doc.page_content[:100]}...")




#COnversation Chain
# Import the ChatMessageHistory class from langchain.memory
from langchain_community.chat_message_histories import ChatMessageHistory

llama_llm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)
# Set up the language model to use for chat interactions
chat = llama_llm

# Create a new conversation history object
# This will store the back-and-forth messages in the conversation
history = ChatMessageHistory()

# Add an initial greeting message from the AI to the history
# This represents a message that would have been sent by the AI assistant
history.add_ai_message("Hello, how can I help you?")
history.add_user_message("What is LangChain used for?")


ai_response = chat.invoke(history.messages)
print("AI:", ai_response)
history.add_ai_message(ai_response)
print(history.messages)


#ConversationBufferMemory
# Import ConversationBufferMemory from langchain.memory module
from langchain.memory import ConversationBufferMemory

# Import ConversationChain from langchain.chains module
from langchain.chains import ConversationChain

# Create a conversation chain with the following components:
conversation = ConversationChain(
    # The language model to use for generating responses
    llm=llama_llm,

    # Set verbose to True to see the full prompt sent to the LLM, including memory contents
    verbose=True,

    # Initialize with ConversationBufferMemory that will:
    # - Store all conversation turns (user inputs and AI responses)
    # - Append the entire conversation history to each new prompt
    # - Provide context for the LLM to generate contextually relevant responses
    memory=ConversationBufferMemory()
)
response = conversation.invoke(input="Hello, I am a little cat. Who are you?")
print("AI:", response['response'])

# Continue conversation
response = conversation.invoke("What can you tell me about LangChain?")
print("AI:", response['response'])