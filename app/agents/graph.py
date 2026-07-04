from langgraph.graph import StateGraph, END
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState

agent = StateGraph(AgentState)

agent.add_node("planner", planner_node)
agent.add_node("retriever", retrieve_node)
agent.add_node("responder", generate_node)

def route_planner(state: AgentState):
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"


agent.set_entry_point("planner")
agent.add_conditional_edges(
    "planner",
    route_planner,
    {"retriever": "retriever", "responder": "responder"},
)
agent.add_edge("retriever", "responder")
agent.add_edge("responder", END)    

checkpointer = MemorySaver()

rag_agent = agent.compile(checkpointer=checkpointer)
