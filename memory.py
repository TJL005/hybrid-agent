import hashlib
import json
import os
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
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        # Write via temp file + rename so a crash mid-write cannot corrupt the cache.
        tmp_path = self.cache_path.with_name(self.cache_path.name + ".tmp")
        tmp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.cache_path)

    def _entry_fresh(self, entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        cached_at = entry.get("cached_at")
        if not isinstance(cached_at, (int, float)) or isinstance(cached_at, bool):
            return False
        return time.time() - cached_at <= self.cache_ttl_seconds

    def get_cached(self, request: str) -> dict[str, Any] | None:
        entry = self._load_cache().get(self._cache_key(request))
        return entry if self._entry_fresh(entry) else None

    def set_cached(self, request: str, decision: dict[str, Any], answer: str | None = None) -> None:
        cache = {k: v for k, v in self._load_cache().items() if self._entry_fresh(v)}
        cache[self._cache_key(request)] = {
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
        summary = "" if result_summary is None else str(result_summary)
        entry = {
            "timestamp": time.time(),
            "request": str(request),
            "route": str(route),
            "runs_used": int(runs_used),
            "actions": [str(action) for action in (actions or [])],
            "result_summary": summary[:500],
        }
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _read_log_entries(self) -> list[dict[str, Any]]:
        if not self.run_log_path.exists():
            return []
        try:
            raw_lines = self.run_log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        entries: list[dict[str, Any]] = []
        for line in raw_lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def recent_context(self, n: int = 5) -> str:
        lines: list[str] = []
        for entry in self._read_log_entries()[-n:]:
            request_text = str(entry.get("request") or "")[:80]
            lines.append(
                f"- [{entry.get('route', '?')}] {request_text} "
                f"({entry.get('runs_used', 0)} runs)"
            )
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        totals: dict[str, int] = {"direct": 0, "single": 0, "pipeline": 0}
        runs_consumed = 0
        total_requests = 0
        for entry in self._read_log_entries():
            total_requests += 1
            route = entry.get("route", "pipeline")
            if not isinstance(route, str):
                route = "pipeline"
            totals[route] = totals.get(route, 0) + 1
            try:
                runs_consumed += int(entry.get("runs_used", 0))
            except (TypeError, ValueError):
                pass
        return {
            "total_requests": total_requests,
            "tiers": totals,
            "runs_consumed": runs_consumed,
        }
