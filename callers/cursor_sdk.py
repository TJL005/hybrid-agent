import os
from typing import Any, Callable

from errors import HybridAgentError

try:
    from cursor_sdk import Agent, AgentOptions, Cursor, CursorAgentError, LocalAgentOptions
except ImportError:  # pragma: no cover - tested via mocks
    Agent = None
    AgentOptions = None
    Cursor = None
    CursorAgentError = Exception
    LocalAgentOptions = None


def _format_model_hint() -> str:
    if Cursor is None:
        return ""
    try:
        models = Cursor.models.list()
        ids = ", ".join(m.id for m in models[:20])
        return f" Available models (sample): {ids}"
    except Exception:
        return ""


def create_cursor_caller(
    *,
    cwd: str | None = None,
    api_key: str | None = None,
    mode: str = "plan",
) -> Callable[[str, str, str], str]:
    """Return a model_caller that routes all stages through the Cursor SDK."""

    if Agent is None:
        raise HybridAgentError("cursor-sdk is not installed. Run: pip install cursor-sdk")

    workdir = cwd or os.getcwd()
    resolved_key = api_key or os.environ.get("CURSOR_API_KEY")

    def cursor_caller(model_name: str, system_prompt: str, user_message: str) -> str:
        prompt = f"<role_instructions>\n{system_prompt}\n</role_instructions>\n\n{user_message}"
        options_kwargs: dict[str, Any] = {
            "model": model_name,
            "mode": mode,
            "local": LocalAgentOptions(cwd=workdir),
        }
        if resolved_key:
            options_kwargs["api_key"] = resolved_key
        options = AgentOptions(**options_kwargs)

        try:
            result = Agent.prompt(prompt, options)
        except CursorAgentError as err:
            hint = _format_model_hint()
            raise HybridAgentError(
                f"Cursor startup failed ({model_name}): {err.message}.{hint}"
            ) from err

        if result.status != "finished":
            raise HybridAgentError(f"Cursor run {result.status} ({model_name}): {result.id}")

        return result.result or ""

    return cursor_caller
