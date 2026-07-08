from unittest.mock import patch

from cli import _parse_duration, main


def test_parse_duration_minutes():
    assert _parse_duration("15m") == 900


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


def test_main_stats():
    with patch("cli.Brain") as mock_brain_cls:
        mock_brain_cls.return_value.memory.stats.return_value = {
            "total_requests": 2,
            "runs_consumed": 3,
            "tiers": {"direct": 1, "single": 0, "pipeline": 1},
        }
        code = main(["stats"])
    assert code == 0
