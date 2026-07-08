import json
import re
from typing import Any, Callable

from capabilities import CapabilityRegistry
from errors import HybridAgentError
from executors import ExecutorContext

DEFAULT_MODEL = "composer-2.5"
MAX_TASKS = 5
MAX_CURSOR_AGENT_TASKS = 2

ADVISOR_SYSTEM = """You are a strategic advisor. Analyze the user's request and produce a concise strategy brief covering:
- Goal and success criteria
- Key constraints and assumptions
- Recommended approach (high level)
Keep it under 500 words. Output plain text only."""

ORCHESTRATOR_SYSTEM_BASE = """You are a task orchestrator. Given a strategy brief, break the work into 1-5 concrete subtasks.
Respond with ONLY a JSON array (no markdown fences) where each item has:
- "id": string (e.g. "task-1")
- "title": short task title
- "prompt": detailed instructions for a worker to complete this task
- "capability": string chosen from the available capabilities list"""

SYNTHESIS_SYSTEM = """You are a synthesis editor. Merge the worker outputs into one cohesive, well-structured final answer in markdown.
Preserve important details. Remove redundancy. Match the tone of the original request."""


def orchestrator_system(registry: CapabilityRegistry) -> str:
    capability_block = registry.descriptions_for_prompt()
    if capability_block:
        return f"{ORCHESTRATOR_SYSTEM_BASE}\n\n{capability_block}"
    return ORCHESTRATOR_SYSTEM_BASE


def parse_task_list(raw: str, registry: CapabilityRegistry | None = None) -> list[dict[str, Any]]:
    """Parse orchestrator output into a task list, tolerating markdown fences and prose."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        tasks = json.loads(text)
    except json.JSONDecodeError as err:
        # Model may wrap the JSON in prose; try the outermost brackets.
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise HybridAgentError(f"orchestrator returned unparseable JSON: {err}") from err
        try:
            tasks = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise HybridAgentError(f"orchestrator returned unparseable JSON: {err}") from err
    if not isinstance(tasks, list):
        raise HybridAgentError("Orchestrator output must be a JSON array")
    if len(tasks) > MAX_TASKS:
        raise HybridAgentError(
            f"Orchestrator returned {len(tasks)} tasks; max is {MAX_TASKS}"
        )
    for task in tasks:
        if not isinstance(task, dict):
            raise HybridAgentError("Each task must be a JSON object")
        if not all(key in task for key in ("id", "title", "prompt")):
            raise HybridAgentError("Each task must have id, title, and prompt")
        capability = task.get("capability") or "llm"
        if not isinstance(capability, str):
            raise HybridAgentError(f"Task capability must be a string, got: {capability!r}")
        if registry is not None and not registry.has(capability):
            raise HybridAgentError(f"Unknown capability: {capability}")
        task["capability"] = capability
    return tasks


class HybridAgent:
    def __init__(
        self,
        advisor_model: str = DEFAULT_MODEL,
        orchestrator_model: str = DEFAULT_MODEL,
        worker_model: str = DEFAULT_MODEL,
        model_caller: Callable[[str, str, str], str] | None = None,
        verbose: bool = False,
        cwd: str | None = None,
        mode: str = "plan",
        api_key: str | None = None,
        registry: CapabilityRegistry | None = None,
        allow_agent_runs: bool = False,
    ):
        self.advisor_model = advisor_model
        self.orchestrator_model = orchestrator_model
        self.worker_model = worker_model
        self.verbose = verbose
        self.cwd = cwd
        self.api_key = api_key
        self.registry = registry or CapabilityRegistry.default()
        self.allow_agent_runs = allow_agent_runs

        if model_caller is None:
            from callers.cursor_sdk import create_cursor_caller

            model_caller = create_cursor_caller(cwd=cwd, api_key=api_key, mode=mode)
        self.model_caller = model_caller
        self._runs_used = 0

    @property
    def runs_used(self) -> int:
        return self._runs_used

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _call(self, stage: str, model: str, system: str, user: str) -> str:
        try:
            self._runs_used += 1
            return self.model_caller(model, system, user)
        except HybridAgentError:
            raise
        except Exception as err:
            raise HybridAgentError(f"{stage} failed: {err}") from err

    def _execute_task(self, strategy: str, task: dict[str, Any]) -> str:
        capability_name = task.get("capability") or "llm"
        try:
            capability = self.registry.get(capability_name)
        except KeyError as err:
            raise HybridAgentError(f"Unknown capability: {capability_name}") from err

        ctx = ExecutorContext(
            strategy=strategy,
            task=task,
            model_caller=self.model_caller,
            worker_model=self.worker_model,
            cwd=self.cwd,
            api_key=self.api_key,
            allow_agent_runs=self.allow_agent_runs,
        )
        self._runs_used += 1
        try:
            return capability.executor(ctx)
        except HybridAgentError:
            raise
        except Exception as err:
            raise HybridAgentError(
                f"task {task.get('id', '?')} ({capability_name}) failed: {err}"
            ) from err

    def process(self, user_message: str, recent_context: str = "") -> str:
        self._runs_used = 0
        advisor_input = user_message
        if recent_context:
            advisor_input = f"Recent context:\n{recent_context}\n\nRequest:\n{user_message}"

        self._log("=== Advisor ===")
        strategy = self._call(
            "advisor",
            self.advisor_model,
            ADVISOR_SYSTEM,
            advisor_input,
        )

        self._log("=== Orchestrator ===")
        orchestrator_raw = self._call(
            "orchestrator",
            self.orchestrator_model,
            orchestrator_system(self.registry),
            f"Strategy brief:\n{strategy}\n\nOriginal request:\n{user_message}",
        )

        tasks = parse_task_list(orchestrator_raw, self.registry)

        if not tasks:
            raise HybridAgentError("orchestrator returned an empty task list")

        cursor_agent_count = sum(1 for t in tasks if t.get("capability") == "cursor_agent")
        if cursor_agent_count > MAX_CURSOR_AGENT_TASKS:
            raise HybridAgentError(
                f"Too many cursor_agent tasks ({cursor_agent_count}); max is {MAX_CURSOR_AGENT_TASKS}"
            )

        self._log(f"=== Workers ({len(tasks)} tasks) ===")
        worker_outputs: list[dict[str, str]] = []
        for task in tasks:
            output = self._execute_task(strategy, task)
            worker_outputs.append(
                {
                    "id": task["id"],
                    "title": task["title"],
                    "capability": task.get("capability", "llm"),
                    "output": output,
                }
            )

        self._log("=== Synthesis ===")
        synthesis_input = json.dumps(
            {"original_request": user_message, "strategy": strategy, "worker_outputs": worker_outputs},
            indent=2,
        )
        return self._call(
            "synthesis",
            self.orchestrator_model,
            SYNTHESIS_SYSTEM,
            synthesis_input,
        )
