import logfire

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.agents.state import AgentState


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)


def planner_node(state: AgentState):
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    # Conversation history (exclude latest user message)
    history_msgs = state["messages"][:-1]
    history = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history_msgs
    )

    system_prompt = """
You are an intent classifier for an Enterprise RAG assistant.

Your job is to determine whether the latest user message requires retrieving enterprise documents.

Return EXACTLY one word:

CONVERSATIONAL
TECHNICAL

Definitions

CONVERSATIONAL
- The latest message can be answered ONLY from the existing conversation.
- Greetings
- Small talk
- "What is my name?"
- "Repeat your previous answer."
- "Explain that again."
- "Summarize your previous response."

TECHNICAL
- The latest message requires enterprise documentation or external technical knowledge.
- Kubernetes
- Intel
- Networking
- APIs
- Configuration
- Troubleshooting
- Documentation lookup
- Any new technical question.

Rules

- Use the conversation history ONLY to determine whether the latest message refers to previous conversation.
- Do NOT classify a question as CONVERSATIONAL simply because the same technical topic appeared earlier.
- If the user asks a new technical question, always return TECHNICAL.
- If you are uncertain, return TECHNICAL.

Return ONLY one word:

CONVERSATIONAL
or
TECHNICAL
"""

    if history:
        system_prompt += f"""

CONVERSATION HISTORY

The following history is provided ONLY to determine whether the
latest user message refers to a previous conversation.

Do NOT classify a new technical question as CONVERSATIONAL
simply because the same topic appeared earlier.

{history}
"""

    classify_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    with logfire.span("Planner - Intent Classification"):

        response = llm.invoke(classify_messages)
        raw = response.content.strip().upper()

        if raw == "CONVERSATIONAL":
            intent = "CONVERSATIONAL"
        elif raw == "TECHNICAL":
            intent = "TECHNICAL"
        else:
            logfire.warning(
                "Unexpected planner output. Defaulting to TECHNICAL.",
                raw_response=raw,
            )
            intent = "TECHNICAL"

        print(f"[Planner] User message : {user_message!r}")
        print(f"[Planner] LLM raw      : {raw!r}")
        print(f"[Planner] Decision     : {intent}")

        logfire.info(
            "Planner decision",
            intent=intent,
            raw_response=raw,
            user_message=user_message,
        )

    if intent == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": [
                "Intent: Conversational",
                "Retrieval: Skipped",
            ],
        }

    return {
        "current_query": user_message,
        "status": f"Technical query detected. Retrieving docs for: {user_message}",
        "plan": [
            "Intent: Technical",
            "Retrieval: Required",
        ],
    }