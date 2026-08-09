from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


# ==========================
# LLM
# ==========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# ==========================
# Embeddings
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================
# Chroma Database
# ==========================

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)