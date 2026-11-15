
import asyncio
import importlib
import os
import sys
import types
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

from src.models.models import AskRequest


@pytest.fixture
def app_module(monkeypatch):
    previous_app_module = sys.modules.pop("app.app", None)
    workflow_module = types.ModuleType("src.agent_graph.graph_workflow")
    workflow_module.graph = Mock()
    monkeypatch.setitem(
        sys.modules, "src.agent_graph.graph_workflow", workflow_module
    )

    module = importlib.import_module("app.app")
    yield module

    sys.modules.pop("app.app", None)
    if previous_app_module is not None:
        sys.modules["app.app"] = previous_app_module


def test_run_agent_passes_complete_state_to_graph(app_module):
    question = "How does the RAG workflow work?"
    graph_result = {
        "answer": "It retrieves and grades evidence.",
        "source_used": "private_kb",
    }
    app_module.graph.invoke.return_value = graph_result

    result = app_module.run_agent(question)

    assert result is graph_result
    app_module.graph.invoke.assert_called_once_with(
        {
            "question": question,
            "current_query": question,
            "kb_docs": [],
            "web_results": "",
            "kb_grade": "",
            "web_grade": "",
            "answer": "",
            "source_used": "",
            "retry_count": 0,
        }
    )


def test_ask_agent_returns_public_response_shape(app_module, monkeypatch):
    question = "How does the RAG workflow work?"
    result = {
        "answer": "It retrieves and grades evidence.",
        "source_used": "private_kb",
    }
    monkeypatch.setattr(app_module, "run_agent", lambda _: result)

    response = asyncio.run(
        app_module.ask_agent(AskRequest(question=question))
    )

    assert response == {
        "question": question,
        "source_used": "private_kb",
        "answer": "It retrieves and grades evidence.",
    }


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


@pytest.fixture
def live_services():
    load_dotenv()
    if os.getenv("RUN_AGENT_INTEGRATION", "").lower() not in {"1", "true"}:
        pytest.skip(
            "Set RUN_AGENT_INTEGRATION=1 to call the configured external services."
        )

    required_keys = ("PINECONE_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY")
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        pytest.fail(
            "Live integration tests require: " + ", ".join(missing_keys)
        )


def assert_rag_chunks(documents):
    assert isinstance(documents, list)
    assert documents, "The retriever returned no chunks."

    contents = [document.page_content.strip() for document in documents]
    assert all(contents), "The retriever returned an empty chunk."

    combined_content = "\n".join(contents).lower()
    assert any(
        term in combined_content
        for term in ("agentic", "retriev", "document", "grading")
    ), "Retrieved chunks do not appear related to Agentic RAG."


def test_live_retriever_returns_rag_chunks(live_services):
    from src.vectordb.load_vectordb import retriever

    query = "How does the Agentic RAG workflow retrieve and grade documents?"
    documents = retriever.invoke(query)

    assert_rag_chunks(documents)


def test_live_app_workflow_retrieves_chunks_and_returns_answer(
    live_services, monkeypatch
):
    from src.agent_graph import nodes
    from app.app import ask_agent

    query = "How does the Agentic RAG workflow retrieve and grade documents?"
    original_retriever = nodes.retriever
    retrieved_chunks = []
    original_reranker = nodes.reranker
    reranked_pairs = []

    class TrackingRetriever:
        def invoke(self, current_query):
            documents = original_retriever.invoke(current_query)
            retrieved_chunks.extend(documents)
            return documents

    class TrackingReranker:
        def predict(self, pairs):
            reranked_pairs.extend(pairs)
            return original_reranker.predict(pairs)

    monkeypatch.setattr(nodes, "retriever", TrackingRetriever())
    monkeypatch.setattr(nodes, "reranker", TrackingReranker())

    response = asyncio.run(ask_agent(AskRequest(question=query)))

    assert_rag_chunks(retrieved_chunks)
    assert reranked_pairs, "The workflow did not rerank the retrieved chunks."
    assert response["question"] == query
    assert response["answer"].strip()
    assert response["source_used"] == "private_kb"
