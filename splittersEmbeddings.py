from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")
document = loader.load()
# Create a CharacterTextSplitter with specific configuration:
# - chunk_size=200: Each chunk will contain approximately 200 characters
# - chunk_overlap=20: Consecutive chunks will overlap by 20 characters to maintain context
# - separator="\n": Text will be split at newline characters when possible
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)
# Print the total number of chunks created
# This shows how many smaller Document objects were generated from the original document(s)
# The number depends on the original document length and the chunk_size setting
print(len(chunks))

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

paper_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
pdf_loader = PyPDFLoader(paper_url)
pdf_document = pdf_loader.load()

web_url = "https://python.langchain.com/v0.2/docs/introduction/"
web_loader = WebBaseLoader(web_url)
web_document = web_loader.load()

splitter_1 = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
splitter_2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])

# Apply both splitters to the PDF document
chunks_1 = splitter_1.split_documents(pdf_document)
chunks_2 = splitter_2.split_documents(pdf_document)

#
# def display_document_stats(docs, name):
#     """Display statistics about a list of document chunks"""
#     total_chunks = len(docs)
#     total_chars = sum(len(doc.page_content) for doc in docs)
#     avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
#
#     # Count unique metadata keys across all documents
#     all_metadata_keys = set()
#     for doc in docs:
#         all_metadata_keys.update(doc.metadata.keys())
#
#     # Print the statistics
#     print(f"\n=== {name} Statistics ===")
#     print(f"Total number of chunks: {total_chunks}")
#     print(f"Average chunk size: {avg_chunk_size:.2f} characters")
#     print(f"Metadata keys preserved: {', '.join(all_metadata_keys)}")
#
#     if docs:
#         print("\nExample chunk:")
#         example_doc = docs[min(5, total_chunks - 1)]  # Get the 5th chunk or the last one if fewer
#         print(f"Content (first 150 chars): {example_doc.page_content[:150]}...")
#         print(f"Metadata: {example_doc.metadata}")
#
#         # Calculate length distribution
#         lengths = [len(doc.page_content) for doc in docs]
#         min_len = min(lengths)
#         max_len = max(lengths)
#         print(f"Min chunk size: {min_len} characters")
#         print(f"Max chunk size: {max_len} characters")
#
#
# # Display stats for both chunk sets
# display_document_stats(chunks_1, "Splitter 1")
# display_document_stats(chunks_2, "Splitter 2")
#
# #from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#
# print(f"Number of chunks: {len(chunks)}")
# texts = [chunk.page_content for chunk in chunks]
# embeddings = embedding_model.embed_documents(texts)
# print(f"Total embeddings: {len(embeddings)}")
# print("First embedding vector:")
# print(" first 10 dimensions: ",embeddings[0][:10])
#
# #vector stores
#
from langchain_chroma import Chroma
#
# # Step 1: Extract texts and metadata
# texts = [chunk.page_content for chunk in chunks_2]
# metadatas = [chunk.metadata for chunk in chunks_2]
#
# # Step 2: Create vector store
# chroma_db = Chroma.from_texts(
#     texts=texts,
#     embedding=embedding_model,
#     metadatas=metadatas,
#     persist_directory="./chroma_db"  # optional: stores on disk
# )
#
# #Query the Vector Store (Semantic Search)
# query = "What is LangChain?"
# results = chroma_db.similarity_search(query, k=3)
#
# for i, doc in enumerate(results):
#     print(f"\nResult {i+1}")
#     print(f"Content:\n{doc.page_content[:200]}...")
#     print(f"Metadata: {doc.metadata}")
#

chroma_db = Chroma(
     persist_directory="./chroma_db",
     embedding_function=embedding_model
 )


#Using Retriever

retriever = chroma_db.as_retriever()
docs = retriever.invoke("Langchain")
# The returned document is the one most semantically similar to "Langchain"
print(docs[0])


#exploring
#hierarchical retrieval using ParentDocumentRetriever

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# Set up two different text splitters for a hierarchical splitting approach:

# 1. Parent splitter creates larger chunks (2000 characters)
# This is used to split documents into larger, more contextually complete sections
parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')

# 2. Child splitter creates smaller chunks (400 characters)
# This is used to split the parent chunks into smaller pieces for more precise retrieval
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')

# Create a Chroma vector store with:
# - A specific collection name "split_parents" for organization
# - The previously configured huggingface embeddings function
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embedding_model
)

# Set up an in-memory storage layer for the parent documents
# This will store the larger chunks that provide context, but won't be directly embedded
store = InMemoryStore()

# Create a ParentDocumentRetriever instance that implements hierarchical document retrieval
retriever = ParentDocumentRetriever(
    # The vector store where child document embeddings will be stored and searched
    # This Chroma instance will contain the embeddings for the smaller chunks
    vectorstore=vectorstore,

    # The document store where parent documents will be stored
    # These larger chunks won't be embedded but will be retrieved by ID when needed
    docstore=store,

    # The splitter used to create small chunks (400 chars) for precise vector search
    # These smaller chunks are embedded and used for similarity matching
    child_splitter=child_splitter,

    # The splitter used to create larger chunks (2000 chars) for better context
    # These parent chunks provide more complete information when retrieved
    parent_splitter=parent_splitter,
)

retriever.add_documents(document)
len(list(store.yield_keys())) #parent

#small chunk
sub_docs = vectorstore.similarity_search("Langchain")
print(sub_docs[0].page_content)

retrieved_docs = retriever.invoke("Langchain")
print(retrieved_docs[0].page_content)

#RetrievalQA
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llama_llm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)
qa = RetrievalQA.from_chain_type(
    llm=llama_llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever(),
    return_source_documents=False
)

query = "What is this paper discussing?"
response = qa.invoke(query)

print("Answer:", response)

#source documents (for traceability):
qa_with_sources = RetrievalQA.from_chain_type(
    llm=llama_llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever(),
    return_source_documents=True
)

response = qa_with_sources.invoke("What is this paper discussing?")
print("Answer:", response["result"])
print("\n--- Sources ---")
for doc in response["source_documents"]:
    print(doc.metadata.get("source"), "| Page:", doc.metadata.get("page"))
    print(doc.page_content[:300], "\n")
