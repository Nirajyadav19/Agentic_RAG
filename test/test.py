
import os
import pytest
from dotenv import load_dotenv

# Load local .env when running tests on your computer.
# GitHub Actions will use GitHub Secrets instead.
load_dotenv()


@pytest.fixture(scope="module")
def pinecone_retriever():

    # Check API key availability
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY is not configured")

    # Import actual Pinecone retriever
    from src.vectordb.load_vectordb import retriever

    return retriever


def test_retriever_returns_documents(pinecone_retriever):

    # Test query
    query = "What information is available in the knowledge base?"

    # Perform actual Pinecone retrieval
    documents = pinecone_retriever.invoke(query)

    # Verify results
    assert documents is not None, "Retriever returned None"

    assert isinstance(documents, list), (
        "Retriever should return a list of documents"
    )

    assert len(documents) > 0, (
        "No documents were retrieved from Pinecone"
    )

    # Verify document content
    for document in documents:

        assert hasattr(document, "page_content")

        assert isinstance(document.page_content, str)

        assert document.page_content.strip() != ""

    print(f"\nRetrieved {len(documents)} documents successfully.")
