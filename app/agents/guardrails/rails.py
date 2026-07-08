import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.agents.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails: LLMRails | None = None

def intialize_rails() -> None:
    global _rails

    rails_llm = ChatGroq(api_key= settings.GROQ_API_KEY, model= "llama-3.1-8b-instant", temperature=0)

    config = RailsConfig.from_content(COLANG_CONTENT, YAML_CONTENT)
    _rails = LLMRails(config, llm=rails_llm)
    logfire.info("NeMo Guardrails intialised (llama-3.1-8b-instant)")

def guard(message: str) -> tuple[bool, str | None]:
    if _rails is None:
        logfire.warning("Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("Guardrails check"):
        result = _rails.generate(messages=[{"role" : "user", "content" : message}])
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("Guardrails passed.")
        return False, None