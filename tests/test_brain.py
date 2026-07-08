import json
from pathlib import Path

import pytest

from brain import Brain
from capabilities import Capability, CapabilityRegistry
from executors import ExecutorContext, create_cursor_agent_capability
from hybrid_agent import HybridAgent
from errors import HybridAgentError


TASKS_JSON = """[
  {"id": "task-1", "title": "Agent task", "prompt": "Do work", "capability": "cursor_agent"}
]"""


PIPELINE_TASKS_JSON = """[
  {"id": "task-1", "title": "Research", "prompt": "Research the market", "capability": "llm"}
]"""


def make_pipeline_caller():
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "routing classifier" in system.lower():
            return json.dumps({"route": "pipeline", "reason": "complex"})
        if "strategic advisor" in system.lower():
            return "Strategy brief"
        if "task orchestrator" in system.lower():
            return PIPELINE_TASKS_JSON
        if "focused worker" in system.lower():
            return "llm output"
        if "synthesis editor" in system.lower():
            return "pipeline result"
        return "unexpected"

    return caller, calls


def test_direct_route_one_call(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return json.dumps({"route": "direct", "answer": "Direct answer", "reason": "simple"})

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    result = brain.run("What is 2+2?")
    assert result == "Direct answer"
    assert len(calls) == 1
    assert calls[0][0] == "composer-2.5"


def test_single_route_two_calls(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "routing classifier" in system.lower():
            return json.dumps(
                {
                    "route": "single",
                    "capability": "llm",
                    "input": "Summarize security posture",
                    "reason": "one action",
                }
            )
        return "single capability output"

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    result = brain.run("Run security review")
    assert result == "single capability output"
    assert len(calls) == 2


def test_cache_skips_router(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return json.dumps({"route": "direct", "answer": "Cached path", "reason": "simple"})

    memory = _memory(tmp_path)
    brain = Brain(model_caller=caller, memory=memory)
    brain.run("repeat me")
    brain.run("repeat me")
    assert len(calls) == 1


def test_fresh_bypasses_cache(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return json.dumps({"route": "direct", "answer": "ok", "reason": "simple"})

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    brain.run("same")
    brain.run("same", fresh=True)
    assert len(calls) == 2


def test_router_model_override(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return json.dumps({"route": "direct", "answer": "ok", "reason": "simple"})

    brain = Brain(model_caller=caller, router_model="glm-5.2", memory=_memory(tmp_path))
    brain.run("test")
    assert calls[0][0] == "glm-5.2"


def test_dispatch_cursor_agent_requires_opt_in():
    registry = CapabilityRegistry()
    registry.register(create_cursor_agent_capability())

    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "strategic advisor" in system.lower():
            return "Strategy brief"
        if "task orchestrator" in system.lower():
            return TASKS_JSON
        return "unused"

    agent = HybridAgent(model_caller=caller, registry=registry, allow_agent_runs=False)
    with pytest.raises(HybridAgentError, match="allow_agent_runs"):
        agent.process("do complex work")


def test_dispatch_cursor_agent_with_opt_in():
    registry = CapabilityRegistry()
    executed: list[str] = []

    def fake_agent(ctx: ExecutorContext) -> str:
        executed.append(ctx.task["title"])
        return "agent output"

    registry.register(
        Capability(name="cursor_agent", description="agent", executor=fake_agent)
    )
    registry.register(
        Capability(
            name="llm",
            description="llm",
            executor=lambda ctx: "llm output",
        )
    )

    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "strategic advisor" in system.lower():
            return "Strategy"
        if "task orchestrator" in system.lower():
            return TASKS_JSON
        if "synthesis editor" in system.lower():
            return "final"
        return "unused"

    agent = HybridAgent(model_caller=caller, registry=registry, allow_agent_runs=True)
    result = agent.process("work")
    assert executed == ["Agent task"]
    assert result == "final"


def test_pipeline_route_runs_full_pipeline(tmp_path: Path):
    caller, calls = make_pipeline_caller()
    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    result = brain.run("Build a multi-part launch plan", fresh=True)
    assert result == "pipeline result"
    # router (1) + advisor + orchestrator + 1 worker + synthesis = 5
    assert len(calls) == 5


def test_direct_route_without_answer_downgrades_to_single_llm(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "routing classifier" in system.lower():
            return json.dumps({"route": "direct", "reason": "forgot the answer"})
        return "recovered answer"

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    result = brain.run("odd request")
    assert result == "recovered answer"
    # router + one llm worker call, not a full pipeline
    assert len(calls) == 2
    assert "focused worker" in calls[1][1].lower()


def test_single_route_unknown_capability_falls_back_to_llm(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "routing classifier" in system.lower():
            return json.dumps(
                {"route": "single", "capability": "hallucinated", "input": "do it"}
            )
        return "llm fallback output"

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    result = brain.run("Run something")
    assert result == "llm fallback output"
    assert len(calls) == 2


def test_single_route_null_input_uses_request(tmp_path: Path):
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "routing classifier" in system.lower():
            return json.dumps({"route": "single", "capability": "llm", "input": None})
        return "output"

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    brain.run("original request text")
    assert "original request text" in calls[1][2]


def test_pipeline_runs_used_counts_workers(tmp_path: Path):
    caller, _ = make_pipeline_caller()
    memory = _memory(tmp_path)
    brain = Brain(model_caller=caller, memory=memory)
    brain.run("Build a multi-part launch plan", fresh=True)
    stats = memory.stats()
    # router + advisor + orchestrator + 1 worker + synthesis
    assert stats["runs_consumed"] == 5


def test_capability_exception_wrapped_as_hybrid_agent_error(tmp_path: Path):
    def caller(model: str, system: str, user: str) -> str:
        if "routing classifier" in system.lower():
            return json.dumps({"route": "single", "capability": "llm", "input": "x"})
        raise RuntimeError("worker exploded")

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    with pytest.raises(HybridAgentError, match="llm"):
        brain.run("Run it")


def test_router_model_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BRAIN_ROUTER_MODEL", "glm-5.2-max")
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return json.dumps({"route": "direct", "answer": "ok", "reason": "simple"})

    brain = Brain(model_caller=caller, memory=_memory(tmp_path))
    brain.run("test env override", fresh=True)
    assert calls[0][0] == "glm-5.2-max"


def _memory(tmp_path: Path):
    from memory import RunMemory

    return RunMemory(home=tmp_path)
