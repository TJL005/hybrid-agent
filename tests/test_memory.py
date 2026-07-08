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


def test_cache_file_not_a_dict_is_ignored(memory: RunMemory):
    memory.cache_path.write_text("[1, 2, 3]")
    assert memory.get_cached("x") is None
    memory.set_cached("x", {"route": "direct"})
    assert memory.get_cached("x") is not None


def test_cache_entry_not_a_dict_is_ignored(memory: RunMemory):
    key = memory._cache_key("x")
    memory.cache_path.write_text(json.dumps({key: "corrupt entry"}))
    assert memory.get_cached("x") is None


def test_cache_entry_bad_timestamp_treated_as_expired(memory: RunMemory):
    key = memory._cache_key("x")
    entry = {"request": "x", "decision": {}, "answer": "a", "cached_at": "yesterday"}
    memory.cache_path.write_text(json.dumps({key: entry}))
    assert memory.get_cached("x") is None


def test_expired_entries_pruned_on_save(memory: RunMemory):
    memory.set_cached("old request", {"route": "direct"})
    old_key = memory._cache_key("old request")
    cache = json.loads(memory.cache_path.read_text())
    cache[old_key]["cached_at"] = time.time() - 7200
    memory.cache_path.write_text(json.dumps(cache))

    memory.set_cached("new request", {"route": "direct"})
    saved = json.loads(memory.cache_path.read_text())
    assert old_key not in saved
    assert memory._cache_key("new request") in saved


def test_append_run_coerces_non_string_summary(memory: RunMemory):
    memory.append_run(request="r", route="direct", runs_used=1, result_summary=None)
    stats = memory.stats()
    assert stats["total_requests"] == 1


def test_log_entries_that_are_not_dicts_skipped(memory: RunMemory):
    memory.run_log_path.write_text('"just a string"\n[1, 2]\n{"route":"direct","runs_used":1}\n')
    stats = memory.stats()
    assert stats["total_requests"] == 1
    assert memory.recent_context() != ""


def test_stats_tolerates_bad_runs_used(memory: RunMemory):
    memory.run_log_path.write_text('{"route":"direct","request":"ok","runs_used":"many"}\n')
    stats = memory.stats()
    assert stats["total_requests"] == 1
    assert stats["runs_consumed"] == 0


def test_recent_context_tolerates_null_request(memory: RunMemory):
    memory.run_log_path.write_text('{"route":"direct","request":null,"runs_used":1}\n')
    assert "direct" in memory.recent_context()
