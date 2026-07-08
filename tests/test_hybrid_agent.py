import json

import pytest

from hybrid_agent import MAX_TASKS, HybridAgent, HybridAgentError, parse_task_list
from capabilities import CapabilityRegistry
from executors import create_llm_capability


TASKS_JSON = """[
  {"id": "task-1", "title": "Research", "prompt": "Research the market"},
  {"id": "task-2", "title": "Draft", "prompt": "Draft the plan"}
]"""


def make_mock_caller():
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        if "strategic advisor" in system.lower():
            return "Strategy: launch in Q3 with beta users."
        if "task orchestrator" in system.lower():
            return TASKS_JSON
        if "focused worker" in system.lower():
            return f"Worker output for: {user[:40]}"
        if "synthesis editor" in system.lower():
            return "# Final Answer\n\nMerged result."
        return "unexpected"

    return caller, calls


def test_process_runs_all_stages():
    caller, calls = make_mock_caller()
    agent = HybridAgent(model_caller=caller)
    result = agent.process("Write a launch plan")

    assert result == "# Final Answer\n\nMerged result."
    assert len(calls) == 5  # advisor, orchestrator, 2 workers, synthesis


def test_orchestrator_json_parsing_clean():
    tasks = parse_task_list(TASKS_JSON)
    assert len(tasks) == 2
    assert tasks[0]["id"] == "task-1"


def test_orchestrator_json_parsing_fenced():
    fenced = f"```json\n{TASKS_JSON}\n```"
    tasks = parse_task_list(fenced)
    assert len(tasks) == 2


def test_orchestrator_json_parsing_unknown_capability():
    registry = CapabilityRegistry()
    registry.register(create_llm_capability())
    bad_json = '[{"id": "1", "title": "T", "prompt": "p", "capability": "missing"}]'
    with pytest.raises(HybridAgentError, match="Unknown capability"):
        parse_task_list(bad_json, registry)


def test_orchestrator_json_parsing_invalid():
    with pytest.raises(HybridAgentError):
        parse_task_list("not json")


def test_orchestrator_json_parsing_with_surrounding_prose():
    raw = f"Here are the tasks:\n{TASKS_JSON}\nHope that helps!"
    tasks = parse_task_list(raw)
    assert len(tasks) == 2


def test_orchestrator_rejects_non_dict_task_items():
    # A bare string would pass a naive `"id" in task` substring check.
    with pytest.raises(HybridAgentError, match="JSON object"):
        parse_task_list('["id title prompt"]')


def test_orchestrator_rejects_too_many_tasks():
    tasks = [{"id": f"t{i}", "title": "T", "prompt": "p"} for i in range(MAX_TASKS + 1)]
    with pytest.raises(HybridAgentError, match="max"):
        parse_task_list(json.dumps(tasks))


def test_null_capability_defaults_to_llm():
    registry = CapabilityRegistry.default()
    raw = '[{"id": "1", "title": "T", "prompt": "p", "capability": null}]'
    tasks = parse_task_list(raw, registry)
    assert tasks[0]["capability"] == "llm"


def test_worker_exception_wrapped_as_hybrid_agent_error():
    def caller(model: str, system: str, user: str) -> str:
        if "strategic advisor" in system.lower():
            return "Strategy"
        if "task orchestrator" in system.lower():
            return TASKS_JSON
        if "focused worker" in system.lower():
            raise RuntimeError("boom")
        return "unexpected"

    agent = HybridAgent(model_caller=caller)
    with pytest.raises(HybridAgentError, match="task-1"):
        agent.process("test")


def test_runs_used_counts_worker_tasks():
    caller, _ = make_mock_caller()
    agent = HybridAgent(model_caller=caller)
    agent.process("Write a launch plan")
    # advisor + orchestrator + 2 workers + synthesis
    assert agent.runs_used == 5


def test_empty_task_list_raises():
    caller, _ = make_mock_caller()

    def empty_orchestrator(model: str, system: str, user: str) -> str:
        if "task orchestrator" in system.lower():
            return "[]"
        return make_mock_caller()[0](model, system, user)

    agent = HybridAgent(model_caller=empty_orchestrator)
    with pytest.raises(HybridAgentError, match="empty task list"):
        agent.process("test")


def test_cheap_agent_models():
    caller, calls = make_mock_caller()
    agent = HybridAgent(
        advisor_model="glm-5.2",
        orchestrator_model="glm-5.2",
        worker_model="composer-2.5",
        model_caller=caller,
    )
    agent.process("Create a marketing strategy")

    models_used = [c[0] for c in calls]
    assert models_used[0] == "glm-5.2"  # advisor
    assert models_used[1] == "glm-5.2"  # orchestrator
    assert models_used[2] == "composer-2.5"  # worker 1
    assert models_used[3] == "composer-2.5"  # worker 2
    assert models_used[4] == "glm-5.2"  # synthesis


def test_verbose_logging(capsys):
    caller, _ = make_mock_caller()
    agent = HybridAgent(model_caller=caller, verbose=True)
    agent.process("test request")

    captured = capsys.readouterr().out
    assert "=== Advisor ===" in captured
    assert "=== Orchestrator ===" in captured
    assert "=== Workers (2 tasks) ===" in captured
    assert "=== Synthesis ===" in captured
