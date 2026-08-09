import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================
# Environment
# =====================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")


# =====================================
# LLM
# =====================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)


# =====================================
# Embeddings
# =====================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================
# Vector Database
# =====================================

DB_PATH = "chroma_db"


if os.path.exists(DB_PATH) and os.listdir(DB_PATH):

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

else:

    loader = DirectoryLoader(
        "data",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        }
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(f"Created {len(chunks)} chunks.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Vector database created successfully.")


# =====================================
# Retriever
# =====================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)