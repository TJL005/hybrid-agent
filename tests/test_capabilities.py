import pytest

from capabilities import CapabilityRegistry
from executors import ExecutorContext, create_llm_capability


def test_register_and_lookup():
    registry = CapabilityRegistry()
    cap = create_llm_capability()
    registry.register(cap)
    assert registry.get("llm").name == "llm"


def test_unknown_capability_raises():
    registry = CapabilityRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")


def test_descriptions_for_prompt():
    registry = CapabilityRegistry()
    registry.register(create_llm_capability())
    text = registry.descriptions_for_prompt()
    assert '"llm"' in text
    assert "language model" in text.lower()


def test_default_registry_has_llm():
    registry = CapabilityRegistry.default()
    assert registry.has("llm")


def test_llm_executor_uses_model_caller():
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return "done"

    ctx = ExecutorContext(
        strategy="strategy",
        task={"id": "1", "title": "T", "prompt": "do it"},
        model_caller=caller,
        worker_model="composer-2.5",
    )
    result = create_llm_capability().executor(ctx)
    assert result == "done"
    assert calls[0][0] == "composer-2.5"
