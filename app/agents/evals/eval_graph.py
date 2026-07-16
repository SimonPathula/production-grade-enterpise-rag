from langgraph.graph import StateGraph, END

from app.agents.evals.eval_state import EvaluationState
from app.agents.evals.nodes.eval_responder import generate_node
from app.agents.evals.nodes.eval_retriever import retrieve_node

builder = StateGraph(EvaluationState)

builder.add_node("retriever", retrieve_node)
builder.add_node("responder", generate_node)

builder.set_entry_point("retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

eval_rag_agent = builder.compile()