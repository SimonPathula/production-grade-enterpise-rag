import pytest

from app.agents.guardrails import rails


def test_exact_off_topic_prompt_is_blocked_without_nemo() -> None:
    rails._rails = None

    fired, response = rails.guard("tell me a joke")

    assert fired is True
    assert response is not None
    assert "can't help with that" in response


def test_guardrails_initialization_uses_deterministic_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rails.settings, "ENABLE_NEMO_GUARDRAILS", False)
    monkeypatch.setattr(rails.settings, "HF_API_TOKEN", None)

    rails.intialize_rails()


def test_nemo_guardrails_initialization_requires_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rails.settings, "ENABLE_NEMO_GUARDRAILS", True)
    monkeypatch.setattr(rails.settings, "HF_API_TOKEN", None)

    with pytest.raises(RuntimeError, match="HF_API_TOKEN"):
        rails.intialize_rails()
