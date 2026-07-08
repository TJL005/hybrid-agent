import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _default_home() -> Path:
    return Path.home() / ".hybrid-agent"


class RunMemory:
    def __init__(self, home: Path | None = None, cache_ttl_seconds: int = 3600) -> None:
        self.home = home or _default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.home / "runs.jsonl"
        self.cache_path = self.home / "cache.json"
        self.cache_ttl_seconds = cache_ttl_seconds

    def _normalize(self, request: str) -> str:
        return " ".join(request.strip().lower().split())

    def _cache_key(self, request: str) -> str:
        normalized = self._normalize(request)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        self.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    def get_cached(self, request: str) -> dict[str, Any] | None:
        cache = self._load_cache()
        entry = cache.get(self._cache_key(request))
        if not entry:
            return None
        if time.time() - entry.get("cached_at", 0) > self.cache_ttl_seconds:
            return None
        return entry

    def set_cached(self, request: str, decision: dict[str, Any], answer: str | None = None) -> None:
        cache = self._load_cache()
        key = self._cache_key(request)
        cache[key] = {
            "request": request,
            "decision": decision,
            "answer": answer,
            "cached_at": time.time(),
        }
        self._save_cache(cache)

    def append_run(
        self,
        *,
        request: str,
        route: str,
        runs_used: int,
        result_summary: str,
        actions: list[str] | None = None,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "request": request,
            "route": route,
            "runs_used": runs_used,
            "actions": actions or [],
            "result_summary": result_summary[:500],
        }
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def recent_context(self, n: int = 5) -> str:
        if not self.run_log_path.exists():
            return ""
        lines: list[str] = []
        try:
            raw_lines = self.run_log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        for line in raw_lines[-n:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            lines.append(
                f"- [{entry.get('route', '?')}] {entry.get('request', '')[:80]} "
                f"({entry.get('runs_used', 0)} runs)"
            )
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        totals: dict[str, int] = {"direct": 0, "single": 0, "pipeline": 0}
        runs_consumed = 0
        total_requests = 0
        if not self.run_log_path.exists():
            return {"total_requests": 0, "tiers": totals, "runs_consumed": 0}
        try:
            raw_lines = self.run_log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {"total_requests": 0, "tiers": totals, "runs_consumed": 0}
        for line in raw_lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_requests += 1
            route = entry.get("route", "pipeline")
            totals[route] = totals.get(route, 0) + 1
            runs_consumed += int(entry.get("runs_used", 0))
        return {
            "total_requests": total_requests,
            "tiers": totals,
            "runs_consumed": runs_consumed,
        }
