#pip install --user "wget==3.2"
import wget
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_ollama import OllamaLLM


filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# Use wget to download the file
wget.download(url, out=filename)
print('file downloaded')

loader = TextLoader(filename)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(len(texts))

embeddings = HuggingFaceEmbeddings()

docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested')

parameters = {
    'DECODING_METHOD': 'greedy',
    'MIN_NEW_TOKENS': 130,
    'MAX_NEW_TOKENS': 256,
    'TEMPERATURE': 0.5
}

# Initialize the OllamaLLM with the parameters
llm = OllamaLLM(
    model="mistral",
    temperature=parameters['TEMPERATURE'],
    max_tokens=parameters['MAX_NEW_TOKENS'],
    min_tokens=parameters['MIN_NEW_TOKENS'],
    decoding_method=parameters['DECODING_METHOD']
)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 return_source_documents=False)
query = "what is mobile policy?"
resp=qa.invoke(query)
print(resp)

prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs,
                                 return_source_documents=False)

query = "Can I eat in company vehicles?"
response=qa.invoke(query)
print(response)

memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)

qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                           chain_type="stuff",
                                           retriever=docsearch.as_retriever(),
                                           memory = memory,
                                           get_chat_history=lambda h : h,
                                           return_source_documents=False)


def qaFn():
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               chain_type="stuff",
                                               retriever=docsearch.as_retriever(),
                                               memory=memory,
                                               get_chat_history=lambda h: h,
                                               return_source_documents=False)
    history = []
    while True:
        query = input("Question: ")

        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break

        result = qaFn({"question": query}, {"chat_history": history})

        history.append((query, result["answer"]))

        print("Answer: ", result["answer"])