import json

import pytest

from capabilities import CapabilityRegistry
from executors import create_llm_capability
from hybrid_agent import HybridAgentError
from router import DEFAULT_ROUTER_MODEL, parse_router_decision, route_request


def test_parse_router_decision_direct():
    raw = json.dumps({"route": "direct", "answer": "Hello!", "reason": "greeting"})
    decision = parse_router_decision(raw)
    assert decision["route"] == "direct"
    assert decision["answer"] == "Hello!"


def test_parse_router_decision_fenced():
    raw = '```json\n{"route": "single", "capability": "llm", "input": "x"}\n```'
    decision = parse_router_decision(raw)
    assert decision["route"] == "single"


def test_invalid_route_defaults_to_pipeline():
    raw = json.dumps({"route": "unknown"})
    decision = parse_router_decision(raw)
    assert decision["route"] == "pipeline"


def test_route_request_calls_router_model_first():
    calls: list[tuple[str, str, str]] = []

    def caller(model: str, system: str, user: str) -> str:
        calls.append((model, system, user))
        return json.dumps({"route": "direct", "answer": "ok", "reason": "simple"})

    registry = CapabilityRegistry()
    registry.register(create_llm_capability())
    decision = route_request(
        "hi",
        model_caller=caller,
        router_model="composer-2.5",
        registry=registry,
    )
    assert decision["route"] == "direct"
    assert len(calls) == 1
    assert calls[0][0] == "composer-2.5"
    assert "routing classifier" in calls[0][1].lower()


def test_malformed_router_falls_back_to_pipeline():
    def caller(model: str, system: str, user: str) -> str:
        return "not json"

    registry = CapabilityRegistry.default()
    decision = route_request(
        "complex multi-step project",
        model_caller=caller,
        router_model=DEFAULT_ROUTER_MODEL,
        registry=registry,
    )
    assert decision["route"] == "pipeline"
