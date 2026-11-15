
import asyncio
import importlib
import os
import sys
import types
from unittest.mock import Mock

import pytest
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


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="Set RUN_AGENT_INTEGRATION=1 to call the configured external services.",
)
def test_live_agent_graph():
    from src.agent_graph.graph_workflow import graph

    question = "Explain how the Agentic RAG workflow retrieves and grades evidence."
    result = graph.invoke(
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

    assert isinstance(result["answer"], str)
    assert result["answer"].strip()
    assert result["source_used"] in {
        "private_kb",
        "web_search",
        "direct",
        "insufficient_evidence",
    }
