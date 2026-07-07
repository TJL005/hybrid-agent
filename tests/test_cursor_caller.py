from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from callers.cursor_sdk import create_cursor_caller
from hybrid_agent import HybridAgentError


class FakeCursorAgentError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@patch("callers.cursor_sdk.CursorAgentError", FakeCursorAgentError)
@patch("callers.cursor_sdk.Agent")
def test_cursor_caller_returns_result(mock_agent):
    mock_agent.prompt.return_value = SimpleNamespace(status="finished", result="hello", id="run-1")

    caller = create_cursor_caller(cwd="/tmp", api_key="test-key")
    result = caller("composer-2.5", "system prompt", "user message")

    assert result == "hello"
    mock_agent.prompt.assert_called_once()
    prompt_arg, options_arg = mock_agent.prompt.call_args[0]
    assert "<role_instructions>" in prompt_arg
    assert "system prompt" in prompt_arg
    assert "user message" in prompt_arg
    assert options_arg.model == "composer-2.5"
    assert options_arg.mode == "plan"
    assert options_arg.api_key == "test-key"


@patch("callers.cursor_sdk.CursorAgentError", FakeCursorAgentError)
@patch("callers.cursor_sdk.Agent")
def test_cursor_caller_maps_run_error(mock_agent):
    mock_agent.prompt.return_value = SimpleNamespace(status="error", result=None, id="run-2")

    caller = create_cursor_caller()
    with pytest.raises(HybridAgentError, match="Cursor run error"):
        caller("composer-2.5", "sys", "user")


@patch("callers.cursor_sdk.CursorAgentError", FakeCursorAgentError)
@patch("callers.cursor_sdk.Agent")
def test_cursor_caller_maps_startup_error(mock_agent):
    mock_agent.prompt.side_effect = FakeCursorAgentError("auth failed")

    caller = create_cursor_caller()
    with pytest.raises(HybridAgentError, match="Cursor startup failed"):
        caller("glm-5.2", "sys", "user")
