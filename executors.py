import os
from dataclasses import dataclass
from typing import Any, Callable

from capabilities import Capability
from errors import HybridAgentError

try:
    from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
except ImportError:  # pragma: no cover
    Agent = None
    AgentOptions = None
    CursorAgentError = Exception
    LocalAgentOptions = None

LLM_WORKER_SYSTEM = (
    "You are a focused worker. Complete the assigned task thoroughly. "
    "Output plain text or markdown."
)


@dataclass
class ExecutorContext:
    strategy: str
    task: dict[str, Any]
    model_caller: Callable[[str, str, str], str]
    worker_model: str
    cwd: str | None = None
    api_key: str | None = None
    allow_agent_runs: bool = False


def llm_executor(ctx: ExecutorContext) -> str:
    return ctx.model_caller(
        ctx.worker_model,
        LLM_WORKER_SYSTEM,
        (
            f"Strategy context:\n{ctx.strategy}\n\n"
            f"Task: {ctx.task['title']}\n\n"
            f"Instructions:\n{ctx.task['prompt']}"
        ),
    )


def cursor_agent_executor(ctx: ExecutorContext) -> str:
    if not ctx.allow_agent_runs:
        raise HybridAgentError(
            "cursor_agent capability requires allow_agent_runs=True on Brain/HybridAgent"
        )
    if Agent is None:
        raise HybridAgentError("cursor-sdk is not installed. Run: pip install cursor-sdk")

    workdir = ctx.cwd or os.getcwd()
    resolved_key = ctx.api_key or os.environ.get("CURSOR_API_KEY")
    prompt = (
        f"Strategy context:\n{ctx.strategy}\n\n"
        f"Task: {ctx.task['title']}\n\n"
        f"Instructions:\n{ctx.task['prompt']}"
    )
    options_kwargs: dict[str, Any] = {
        "model": ctx.worker_model,
        "mode": "agent",
        "local": LocalAgentOptions(cwd=workdir),
    }
    if resolved_key:
        options_kwargs["api_key"] = resolved_key

    try:
        result = Agent.prompt(prompt, AgentOptions(**options_kwargs))
    except CursorAgentError as err:
        raise HybridAgentError(f"cursor_agent startup failed: {err.message}") from err

    if result.status != "finished":
        raise HybridAgentError(f"cursor_agent run {result.status}: {result.id}")

    return result.result or ""


def create_llm_capability() -> Capability:
    return Capability(
        name="llm",
        description="Generate text or analysis via a language model worker.",
        executor=llm_executor,
    )


def create_cursor_agent_capability() -> Capability:
    return Capability(
        name="cursor_agent",
        description=(
            "Run a full Cursor SDK agent against the repo (can edit files, use skills). "
            "Requires allow_agent_runs=True."
        ),
        executor=cursor_agent_executor,
    )
