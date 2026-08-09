from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os


# ==========================
# Load Documents
# ==========================

loader = DirectoryLoader(
    path="data",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)


documents = loader.load()


print(f"Loaded {len(documents)} documents.")



# ==========================
# Add Source Metadata
# ==========================

for doc in documents:

    filename = os.path.basename(
        doc.metadata["source"]
    )

    doc.metadata["source"] = filename



# ==========================
# Split Documents
# ==========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


chunks = text_splitter.split_documents(
    documents
)


print(
    f"Created {len(chunks)} chunks."
)



# ==========================
# Create Embeddings
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# ==========================
# Create Chroma DB
# ==========================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


print(
    "✅ Vector database created successfully!"
)