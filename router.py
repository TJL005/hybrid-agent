import json
import re
from typing import Any, Callable

from capabilities import CapabilityRegistry
from errors import HybridAgentError

DEFAULT_ROUTER_MODEL = "composer-2.5"

ROUTER_SYSTEM = """You are a routing classifier for a personal agent brain.
Your job is to choose the cheapest path that still satisfies the request.

Respond with ONLY a JSON object (no markdown fences) with these fields:
- "route": one of "direct", "single", "pipeline"
- "answer": string (required when route is "direct"; answer the user inline)
- "capability": string (required when route is "single"; must be from the available list)
- "input": string (required when route is "single"; what to pass to that capability)
- "reason": short string explaining your choice

Routing rules (bias toward the cheapest tier):
- "direct": simple questions, greetings, lookups, or anything answerable in one reply. DEFAULT CHOICE.
- "single": one clear action that maps to exactly one capability (e.g. "run security review").
- "pipeline": ONLY when the request genuinely needs multiple distinct work products planned together.
  Do NOT choose pipeline for simple or single-action requests."""


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise HybridAgentError(f"router returned unparseable JSON: {err}") from err
    if not isinstance(data, dict):
        raise HybridAgentError("router output must be a JSON object")
    return data


def parse_router_decision(raw: str) -> dict[str, Any]:
    data = _extract_json(raw)
    route = data.get("route", "pipeline")
    if route not in ("direct", "single", "pipeline"):
        route = "pipeline"
    data["route"] = route
    return data


def route_request(
    request: str,
    *,
    model_caller: Callable[[str, str, str], str],
    router_model: str,
    registry: CapabilityRegistry,
    recent_context: str = "",
) -> dict[str, Any]:
    capability_block = registry.descriptions_for_prompt()
    user_message = request
    if recent_context:
        user_message = f"Recent context:\n{recent_context}\n\nRequest:\n{request}"
    if capability_block:
        user_message = f"{capability_block}\n\n{user_message}"

    try:
        raw = model_caller(router_model, ROUTER_SYSTEM, user_message)
        return parse_router_decision(raw)
    except HybridAgentError:
        return {"route": "pipeline", "reason": "router failed; escalating to pipeline"}
