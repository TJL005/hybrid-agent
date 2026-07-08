import json
import time
from pathlib import Path

import pytest

from memory import RunMemory


@pytest.fixture
def memory(tmp_path: Path) -> RunMemory:
    return RunMemory(home=tmp_path, cache_ttl_seconds=60)


def test_append_run_and_recent_context(memory: RunMemory):
    memory.append_run(
        request="hello",
        route="direct",
        runs_used=1,
        result_summary="hi",
    )
    context = memory.recent_context()
    assert "hello" in context
    assert "direct" in context


def test_cache_hit_and_expiry(memory: RunMemory):
    memory.set_cached("Run security review", {"route": "single", "capability": "llm"})
    cached = memory.get_cached("  run   security review ")
    assert cached is not None
    assert cached["decision"]["route"] == "single"

    memory.cache_ttl_seconds = 0
    expired = memory.get_cached("Run security review")
    assert expired is None


def test_stats_aggregation(memory: RunMemory):
    memory.append_run(request="a", route="direct", runs_used=1, result_summary="x")
    memory.append_run(request="b", route="pipeline", runs_used=5, result_summary="y")
    stats = memory.stats()
    assert stats["total_requests"] == 2
    assert stats["tiers"]["direct"] == 1
    assert stats["tiers"]["pipeline"] == 1
    assert stats["runs_consumed"] == 6


def test_corrupt_log_lines_skipped(memory: RunMemory):
    memory.run_log_path.write_text('{"route":"direct","request":"ok","runs_used":1}\nnot json\n')
    assert memory.recent_context() != ""
    stats = memory.stats()
    assert stats["total_requests"] == 1
