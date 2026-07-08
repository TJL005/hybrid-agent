from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    executor: Callable[..., str]


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Capability:
        if name not in self._capabilities:
            raise KeyError(name)
        return self._capabilities[name]

    def has(self, name: str) -> bool:
        return name in self._capabilities

    def names(self) -> list[str]:
        return sorted(self._capabilities.keys())

    def descriptions_for_prompt(self) -> str:
        if not self._capabilities:
            return ""
        lines = ["Available capabilities:"]
        for cap in sorted(self._capabilities.values(), key=lambda c: c.name):
            lines.append(f'- "{cap.name}": {cap.description}')
        return "\n".join(lines)

    @classmethod
    def default(cls) -> "CapabilityRegistry":
        from executors import create_llm_capability

        registry = cls()
        registry.register(create_llm_capability())
        return registry
