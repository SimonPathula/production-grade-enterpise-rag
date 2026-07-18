from nemoguardrails import LLMRails, RailsConfig
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
from app.agents.guardrails.rails import intialize_rails, guard
from app.agents.evals.eval_graph import eval_rag_agent

app = FastAPI(title="Enterprise Agentic RAG API")

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_user"

@app.on_event("startup")
def startup_event():
    intialize_rails()

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
async def query(request: QueryRequest):
    import asyncio
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
        # Both guard() and rag_agent.invoke() use sync blocking calls — run in thread pool
        rail_fired, rail_response = await asyncio.to_thread(guard, q)
        if rail_fired:
            logfire.info(f"Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
            }

        # Run the blocking LangGraph pipeline in a thread pool so uvicorn stays responsive
        final_output = await asyncio.to_thread(rag_agent.invoke, initial_state, config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }

    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error. Please try again.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }

@app.post("/evaluate")
async def evaluate(request: QueryRequest):

    initial_state = {
        "current_query": request.query,
        "documents": [],
        "raw_documents": [],
        "final_answer": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "retrieval_time_ms": 0.0,
        "rerank_time_ms": 0.0,
        "generation_time_ms": 0.0,
    }

    result = eval_rag_agent.invoke(initial_state)

    return {
        "answer": result["final_answer"],
        "reranked_chunks": result["documents"],
        "raw_chunks": result["raw_documents"],
        "usage": {
            "input_tokens": result["prompt_tokens"],
            "output_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
        "timings_ms": {
            "retrieval": result["retrieval_time_ms"],
            "rerank": result["rerank_time_ms"],
            "generation": result["generation_time_ms"],
        },
    }