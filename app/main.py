from app.config import settings
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=settings.LOGFIRE_TOKEN)

from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Enterprise Agentic RAG API")

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_user"

@app.get("/")
def health():
    return {"status" : "healthy", "message" : "Enterprise LangGraph RAG API is LIVE!!"}

@app.get("/graph")
def get_graph_image():
    try:
        png = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png, media_type="image/png")
    except Exception as e:
        return {"error" : f"Could not generate graph image: {e}"}

@app.post("/query")
def query(request: QueryRequest):
    q = request.query
    thread_id = request.thread_id

    initial_state = {
        "messages" : [{"role" : "user", "content" : q}],
        "current_query" : q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }

    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error. Please try again.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }