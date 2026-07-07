import logfire
from app.config import settings
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_qdrant_db
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
        query = state["current_query"]

        with logfire.span("Documents Retrieval"):
            logfire.info(f"Searching Qdrant for: {query}")
            raw_results = search_qdrant_db(query, 15)
            logfire.info(f"Retrieved {len(raw_results)} candidates from Qdrant DB")

            doc_contents = [doc["content"] for doc in raw_results]

            with logfire.span("Semantic Reranking"):
                reranked_docs = rerank_documents(query, doc_contents)
                logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

            formatted_docs = [f"CONTENT: {doc}" for doc in reranked_docs]

        return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }