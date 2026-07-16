import logfire
from langchain_groq import ChatGroq
from app.agents.evals.eval_state import EvaluationState
from app.config import settings

llm = ChatGroq(model= settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0, max_tokens= 300)

def generate_node(state: EvaluationState):
    query = state["current_query"]
    
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

    USER QUESTION:
    "{query}"
    """
    with logfire.span("LLM Synthesis"):
        try:
            response = llm.invoke(prompt)
            logfire.info("Response synthesized successfully")
            return{
                "final_answer":response.content, 
                "documents" : state["documents"],
                "prompt_tokens": response.usage_metadata["input_tokens"],
                "completion_tokens" : response.usage_metadata["output_tokens"],
                "total_tokens" : response.usage_metadata["total_tokens"]
            }

        except Exception as e:
            logfire.error(f"LLM generation failed: {e}")
            raise e