import os
import streamlit as st
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Cache directory — model downloads here ONCE, never again ──
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── @st.cache_resource — model loads ONCE per app session ──
# Even if you upload 10 reports, model only loads 1 time
@st.cache_resource(show_spinner="⏳ Loading embedding model (first time only)...")
def get_embeddings():
    """Load embedding model once and cache it forever."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=CACHE_DIR,           # saves to disk — survives restarts
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def create_vector_store(chunks: list):
    """Convert text chunks into embeddings and store in FAISS."""

    # Filter empty chunks
    documents = [Document(page_content=c) for c in chunks if c.strip()]

    # Get cached embeddings (loads only once)
    embeddings = get_embeddings()

    # Build FAISS index
    return FAISS.from_documents(documents, embeddings)


def search_vector_store(vector_store, query: str, k: int = 4) -> str:
    """Return top-k relevant chunks for a given query."""
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])