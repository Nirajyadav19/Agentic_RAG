from fastapi import FastAPI
from src.agent_graph.graph_workflow import graph
from src.models.models import AgentState, AskRequest

app = FastAPI(title="Agentic RAG API")


def run_agent(question: str):
    initial_state: AgentState = {
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

    result = graph.invoke(initial_state)

    print("\n" + "=" * 90)
    print("QUESTION:")
    print(question)
    print("\nSOURCE USED:")
    print(result["source_used"])
    print("\nFINAL ANSWER:")
    print(result["answer"])
    print("=" * 90)

    return result


@app.post("/ask")
async def ask_agent(request: AskRequest):
    result = run_agent(request.question)
    return {
        "question": request.question,
        "source_used": result["source_used"],
        "answer": result["answer"],
    }