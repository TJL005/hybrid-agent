import argparse
import sys
from unittest.mock import patch

import pytest

from cli import _parse_duration, main
from errors import HybridAgentError


def test_parse_duration_minutes():
    assert _parse_duration("15m") == 900


def test_parse_duration_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_duration("0s")


def test_parse_duration_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_duration("soon")


def test_main_run_invokes_brain(capsys):
    with patch("cli.Brain") as mock_brain_cls:
        mock_brain = mock_brain_cls.return_value
        mock_brain.run.return_value = "done"
        code = main(["hello world"])
    assert code == 0
    mock_brain.run.assert_called_once_with("hello world", fresh=False)
    assert "done" in capsys.readouterr().out


def test_main_fresh_flag():
    with patch("cli.Brain") as mock_brain_cls:
        mock_brain = mock_brain_cls.return_value
        mock_brain.run.return_value = "ok"
        main(["--fresh", "hello"])
    mock_brain_cls.return_value.run.assert_called_once_with("hello", fresh=True)


def test_main_stats_does_not_construct_brain(capsys):
    with patch("cli.Brain") as mock_brain_cls, patch("cli.RunMemory") as mock_memory_cls:
        mock_memory_cls.return_value.stats.return_value = {
            "total_requests": 2,
            "runs_consumed": 3,
            "tiers": {"direct": 1, "single": 0, "pipeline": 1},
        }
        code = main(["stats"])
    assert code == 0
    mock_brain_cls.assert_not_called()
    assert "Total requests: 2" in capsys.readouterr().out


def test_main_run_handles_brain_construction_error(capsys):
    with patch("cli.Brain", side_effect=HybridAgentError("cursor-sdk is not installed")):
        code = main(["hello"])
    assert code == 1
    assert "Brain error" in capsys.readouterr().err


def test_main_loop_handles_brain_construction_error(capsys):
    with patch("cli.Brain", side_effect=HybridAgentError("cursor-sdk is not installed")):
        code = main(["loop", "check", "--every", "1s", "--max-runs", "1"])
    assert code == 1
    assert "Brain error" in capsys.readouterr().err


def test_main_loop_max_runs(monkeypatch):
    run_count = {"n": 0}

    class FakeBrain:
        def __init__(self, **kwargs):
            pass

        def run(self, request, fresh=False):
            run_count["n"] += 1
            return "loop ok"

    monkeypatch.setattr("cli.Brain", FakeBrain)
    monkeypatch.setattr("cli.time.sleep", lambda _: None)

    code = main(["loop", "check status", "--every", "1s", "--max-runs", "3"])
    assert code == 0
    assert run_count["n"] == 3


def test_main_loop_rejects_negative_max_runs(capsys):
    with pytest.raises(SystemExit):
        main(["loop", "check", "--every", "1s", "--max-runs", "-1"])


def test_main_models_without_sdk(monkeypatch, capsys):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "cursor_sdk", None)
    code = main(["models"])
    assert code == 1
    assert "cursor-sdk" in capsys.readouterr().err
