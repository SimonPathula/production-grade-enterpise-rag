import logfire
from langchain_groq import ChatGroq
from app.agents.state import AgentState
from app.config import settings

llm = ChatGroq(model= settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)

def generate_node(state: AgentState):
    query = state["current_query"]
    history = ""

    for msg in state["messages"]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")

        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history}

        USER QUESTION:
        "{user_msg}"
        """
    with logfire.span("LLM Synthesis"):
        try:
            response = llm.invoke(prompt)
            logfire.info("Response synthesized successfully")
            return{
                "final_answer":response.content, 
                "status": "Response generated.",
                "messages":[{"role" : "assistant", "content": response.content}]
            }

        except Exception as e:
            logfire.error(f"LLM generation failed: {e}")
            raise e