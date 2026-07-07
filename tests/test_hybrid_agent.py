import io
from unittest.mock import patch

import pytest

from hybrid_agent import HybridAgent, HybridAgentError, parse_task_list


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


def test_orchestrator_json_parsing_invalid():
    with pytest.raises(HybridAgentError):
        parse_task_list("not json")


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
