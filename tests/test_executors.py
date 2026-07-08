from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executors import ExecutorContext, cursor_agent_executor
from hybrid_agent import HybridAgentError


class FakeCursorAgentError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def test_cursor_agent_requires_opt_in():
    ctx = ExecutorContext(
        strategy="s",
        task={"id": "1", "title": "T", "prompt": "p"},
        model_caller=lambda *a: "x",
        worker_model="composer-2.5",
        allow_agent_runs=False,
    )
    with pytest.raises(HybridAgentError, match="allow_agent_runs"):
        cursor_agent_executor(ctx)


@patch("executors.CursorAgentError", FakeCursorAgentError)
@patch("executors.Agent")
def test_cursor_agent_executor_success(mock_agent):
    mock_agent.prompt.return_value = SimpleNamespace(status="finished", result="agent done", id="r1")
    ctx = ExecutorContext(
        strategy="s",
        task={"id": "1", "title": "T", "prompt": "fix bug"},
        model_caller=lambda *a: "x",
        worker_model="composer-2.5",
        cwd="/tmp",
        api_key="key",
        allow_agent_runs=True,
    )
    result = cursor_agent_executor(ctx)
    assert result == "agent done"
    options = mock_agent.prompt.call_args[0][1]
    assert options.mode == "agent"


@patch("executors.CursorAgentError", FakeCursorAgentError)
@patch("executors.Agent")
def test_cursor_agent_maps_startup_error(mock_agent):
    mock_agent.prompt.side_effect = FakeCursorAgentError("auth failed")
    ctx = ExecutorContext(
        strategy="s",
        task={"id": "1", "title": "T", "prompt": "p"},
        model_caller=lambda *a: "x",
        worker_model="composer-2.5",
        allow_agent_runs=True,
    )
    with pytest.raises(HybridAgentError, match="cursor_agent startup failed"):
        cursor_agent_executor(ctx)
