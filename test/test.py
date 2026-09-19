import pytest
from src.vectordb.load_vectordb import retriever

def test_retrieval_top_k():
    docs = retriever.invoke(
        "What is RAG?",
    )

    assert len(docs) <= 5