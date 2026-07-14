from langgraph.graph import StateGraph, END
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
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

def get_checkpointer():
    from app.config import settings

    if settings.LOCAL_MODE:
        print("Using local memory saver RAM ")
        return MemorySaver()

    try:
        conninfo = f"postgresql://{settings.DB_USER}:{settings.DB_PASS}@/{settings.DB_NAME}?host=/cloudsql/{settings.DB_CONNECTION_NAME}"
        pool = ConnectionPool(conninfo= conninfo, max_size= 10)

        with pool.connection() as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()

        print("Using persistent PostgresSaver (Cloud SQL Pool)")
        return PostgresSaver(pool)

    except Exception as e:
        print(f"Postgres Connection failed: {e}. Falling back to MemorySaver")
        return MemorySaver()
        
checkpointer = get_checkpointer()

rag_agent = agent.compile(checkpointer=checkpointer)
