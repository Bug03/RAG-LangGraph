from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tiktoken import encoding_for_model

load_dotenv()

# Set up tokenizer for "text-embedding-ada-002"
tokenizer = encoding_for_model("text-embedding-ada-002")

# URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the specified URLs
docs = [WebBaseLoader(url).load() for url in urls]

# In ra số lượng tài liệu tải được từ mỗi URL
for i, url in enumerate(urls):
    print(f"Loaded documents from {url}: {len(docs[i])} documents")

# Print the content of each document
# for i, doc_list in enumerate(docs):
#     print(f"\nDocuments from {urls[i]}:")
#     for j, doc in enumerate(doc_list):
#         print(f"Document {j + 1} content:\n{doc.page_content}\n")



docs_list = [item for sublist in docs for item in sublist]

# Split documents with a chunk size of 350
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=350,  # Set chunk size to 350
    chunk_overlap=0
)

# Split documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Token limit per chunk (using the larger token limit of the model)
max_tokens = 8191  # Max tokens for "text-embedding-ada-002"

# Ensure chunks do not exceed max token limit
filtered_doc_splits = []
for chunk in doc_splits:
    tokens = tokenizer.encode(chunk.page_content)
    if len(tokens) <= max_tokens:
        filtered_doc_splits.append(chunk)

# Batch processing for embedding creation
MAX_BATCH_SIZE = 20  # Define a manageable batch size

batches = [
    filtered_doc_splits[i : i + MAX_BATCH_SIZE] for i in range(0, len(filtered_doc_splits), MAX_BATCH_SIZE)
]

vectorstore = None  # Initialize vectorstore

for batch in batches:
    # Create a Chroma vector store for each batch
    vectorstore = Chroma.from_documents(
        documents=batch,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory="./.chroma"
    )

# Initialize a Chroma retriever
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
).as_retriever()
