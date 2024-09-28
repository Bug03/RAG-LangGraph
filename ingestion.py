from dotenv import load_dotenv

# recursive character text splitter to split up our documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

# We're going to use web based loader to load the documents from the internet
from langchain_community.document_loaders import WebBaseLoader

# Use Chroma as our vector store
from langchain_chroma import Chroma

# import OpenAi embeddings for the embeddings module
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# List of URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the specified URLs
docs = [WebBaseLoader(url).load() for url in urls]

# Flatten the list of documents
docs_list = [item for sublist in docs for item in sublist]

# Initialize the text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Create a Chroma vector store from the split documents
# run once time to create the vector store
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,  # List of document chunks
#     collection_name="rag-chroma",  # Name of the collection in the vector store
#     embedding=OpenAIEmbeddings(),  # Embedding function to convert text to vectors
#     persist_directory="./.chroma", # Directory to persist the vector store
# )


# Initialize a Chroma retriever
retriever = Chroma(
    collection_name="rag-chroma",  # Name of the collection in the vector store
    persist_directory="./.chroma",  # Directory where the vector store is persisted
    embedding_function=OpenAIEmbeddings(),  # Function to convert text to vectors
).as_retriever()  # Convert the Chroma instance to a retriever
