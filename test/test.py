from langchain.agents import AgentState
from src.agent_graph.graph_workflow import graph
from src.vectordb.load_vectordb import retriever

def test_retrieval_top_k():
    docs = retriever.invoke(
        "What is the refund period?"
    )

    assert len(docs) <= 5