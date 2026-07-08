import os
from typing import Any, Callable

from capabilities import CapabilityRegistry
from executors import ExecutorContext, create_cursor_agent_capability, create_llm_capability
from errors import HybridAgentError
from hybrid_agent import HybridAgent
from memory import RunMemory
from router import DEFAULT_ROUTER_MODEL, route_request


def _resolve_router_model(router_model: str | None) -> str:
    return router_model or os.environ.get("BRAIN_ROUTER_MODEL", DEFAULT_ROUTER_MODEL)


class Brain:
    def __init__(
        self,
        *,
        router_model: str | None = None,
        advisor_model: str | None = None,
        orchestrator_model: str | None = None,
        worker_model: str | None = None,
        model_caller: Callable[[str, str, str], str] | None = None,
        registry: CapabilityRegistry | None = None,
        memory: RunMemory | None = None,
        verbose: bool = False,
        cwd: str | None = None,
        api_key: str | None = None,
        allow_agent_runs: bool = False,
    ):
        self.router_model = _resolve_router_model(router_model)
        self.verbose = verbose
        self.cwd = cwd
        self.api_key = api_key
        self.allow_agent_runs = allow_agent_runs
        self.memory = memory or RunMemory()
        self.registry = registry or self._default_registry(allow_agent_runs)
        self.model_caller = model_caller

        if model_caller is None:
            from callers.cursor_sdk import create_cursor_caller

            self.model_caller = create_cursor_caller(cwd=cwd, api_key=api_key, mode="plan")

        self.agent = HybridAgent(
            advisor_model=advisor_model or "composer-2.5",
            orchestrator_model=orchestrator_model or "composer-2.5",
            worker_model=worker_model or "composer-2.5",
            model_caller=self.model_caller,
            verbose=verbose,
            cwd=cwd,
            api_key=api_key,
            registry=self.registry,
            allow_agent_runs=allow_agent_runs,
        )

    @staticmethod
    def _default_registry(allow_agent_runs: bool) -> CapabilityRegistry:
        registry = CapabilityRegistry()
        registry.register(create_llm_capability())
        if allow_agent_runs:
            registry.register(create_cursor_agent_capability())
        return registry

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _execute_capability(self, capability_name: str, input_text: str) -> str:
        if not self.registry.has(capability_name):
            raise HybridAgentError(f"Unknown capability: {capability_name}")

        task = {
            "id": "single-1",
            "title": capability_name,
            "prompt": input_text,
            "capability": capability_name,
        }
        capability = self.registry.get(capability_name)
        ctx = ExecutorContext(
            strategy="",
            task=task,
            model_caller=self.model_caller,
            worker_model=self.agent.worker_model,
            cwd=self.cwd,
            api_key=self.api_key,
            allow_agent_runs=self.allow_agent_runs,
        )
        try:
            return capability.executor(ctx)
        except HybridAgentError:
            raise
        except Exception as err:
            raise HybridAgentError(f"capability '{capability_name}' failed: {err}") from err

    def run(self, request: str, fresh: bool = False) -> str:
        runs_used = 0
        actions: list[str] = []
        recent = self.memory.recent_context()

        decision: dict[str, Any] | None = None
        if not fresh:
            cached = self.memory.get_cached(request)
            if cached:
                answer = cached.get("answer")
                if isinstance(answer, str) and answer.strip():
                    cached_decision = cached.get("decision") or {}
                    self.memory.append_run(
                        request=request,
                        route=cached_decision.get("route", "direct"),
                        runs_used=0,
                        result_summary=answer,
                        actions=["cache_hit"],
                    )
                    return answer
                if isinstance(cached.get("decision"), dict):
                    decision = cached["decision"]
                    self._log(f"=== Router (cached: {decision.get('route')}) ===")

        if decision is None:
            self._log("=== Router ===")
            decision = route_request(
                request,
                model_caller=self.model_caller,
                router_model=self.router_model,
                registry=self.registry,
                recent_context=recent,
            )
            runs_used += 1
            self.memory.set_cached(request, decision)

        route = decision.get("route", "pipeline")

        if route == "direct":
            answer = decision.get("answer")
            if isinstance(answer, str) and answer.strip():
                self.memory.set_cached(request, decision, answer=answer)
                self.memory.append_run(
                    request=request,
                    route="direct",
                    runs_used=runs_used,
                    result_summary=answer,
                )
                return answer
            # Router promised an inline answer but did not provide a usable one.
            self._log("Router chose direct without an answer; downgrading to single llm")
            route = "single"
            decision = {**decision, "capability": "llm", "input": request}

        if route == "single":
            capability_name = decision.get("capability")
            if not isinstance(capability_name, str) or not self.registry.has(capability_name):
                capability_name = "llm" if self.registry.has("llm") else None
            if capability_name is not None:
                input_text = decision.get("input")
                if not isinstance(input_text, str) or not input_text.strip():
                    input_text = request
                actions.append(f"single:{capability_name}")
                result = self._execute_capability(capability_name, input_text)
                runs_used += 1
                self.memory.append_run(
                    request=request,
                    route="single",
                    runs_used=runs_used,
                    result_summary=result,
                    actions=actions,
                )
                return result
            self._log("No usable capability for single route; escalating to pipeline")

        self._log("=== Pipeline ===")
        actions.append("pipeline")
        result = self.agent.process(request, recent_context=recent)
        runs_used += self.agent.runs_used
        self.memory.append_run(
            request=request,
            route="pipeline",
            runs_used=runs_used,
            result_summary=result,
            actions=actions,
        )
        return result
