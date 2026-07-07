import json
import re
from typing import Any, Callable

DEFAULT_MODEL = "composer-2.5"

ADVISOR_SYSTEM = """You are a strategic advisor. Analyze the user's request and produce a concise strategy brief covering:
- Goal and success criteria
- Key constraints and assumptions
- Recommended approach (high level)
Keep it under 500 words. Output plain text only."""

ORCHESTRATOR_SYSTEM = """You are a task orchestrator. Given a strategy brief, break the work into 1-5 concrete subtasks.
Respond with ONLY a JSON array (no markdown fences) where each item has:
- "id": string (e.g. "task-1")
- "title": short task title
- "prompt": detailed instructions for a worker to complete this task"""

SYNTHESIS_SYSTEM = """You are a synthesis editor. Merge the worker outputs into one cohesive, well-structured final answer in markdown.
Preserve important details. Remove redundancy. Match the tone of the original request."""


class HybridAgentError(Exception):
    """Raised when the hybrid agent pipeline fails."""


def parse_task_list(raw: str) -> list[dict[str, Any]]:
    """Parse orchestrator output into a task list, tolerating markdown fences."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        tasks = json.loads(text)
    except json.JSONDecodeError as err:
        raise HybridAgentError(f"orchestrator returned unparseable JSON: {err}") from err
    if not isinstance(tasks, list):
        raise HybridAgentError("Orchestrator output must be a JSON array")
    for task in tasks:
        if not all(key in task for key in ("id", "title", "prompt")):
            raise HybridAgentError("Each task must have id, title, and prompt")
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
    ):
        self.advisor_model = advisor_model
        self.orchestrator_model = orchestrator_model
        self.worker_model = worker_model
        self.verbose = verbose

        if model_caller is None:
            from callers.cursor_sdk import create_cursor_caller

            model_caller = create_cursor_caller(cwd=cwd, api_key=api_key, mode=mode)
        self.model_caller = model_caller

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _call(self, stage: str, model: str, system: str, user: str) -> str:
        try:
            return self.model_caller(model, system, user)
        except HybridAgentError:
            raise
        except Exception as err:
            raise HybridAgentError(f"{stage} failed: {err}") from err

    def process(self, user_message: str) -> str:
        self._log("=== Advisor ===")
        strategy = self._call(
            "advisor",
            self.advisor_model,
            ADVISOR_SYSTEM,
            user_message,
        )

        self._log("=== Orchestrator ===")
        orchestrator_raw = self._call(
            "orchestrator",
            self.orchestrator_model,
            ORCHESTRATOR_SYSTEM,
            f"Strategy brief:\n{strategy}\n\nOriginal request:\n{user_message}",
        )

        try:
            tasks = parse_task_list(orchestrator_raw)
        except (json.JSONDecodeError, HybridAgentError) as err:
            raise HybridAgentError(f"orchestrator returned unparseable JSON: {err}") from err

        if not tasks:
            raise HybridAgentError("orchestrator returned an empty task list")

        self._log(f"=== Workers ({len(tasks)} tasks) ===")
        worker_outputs: list[dict[str, str]] = []
        for task in tasks:
            output = self._call(
                "worker",
                self.worker_model,
                "You are a focused worker. Complete the assigned task thoroughly. Output plain text or markdown.",
                f"Strategy context:\n{strategy}\n\nTask: {task['title']}\n\nInstructions:\n{task['prompt']}",
            )
            worker_outputs.append({"id": task["id"], "title": task["title"], "output": output})

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
