import argparse
import os
import re
import sys
import time

from brain import Brain
from errors import HybridAgentError


def _parse_duration(value: str) -> int:
    match = re.fullmatch(r"(\d+)(s|m|h)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("Duration must look like 30s, 15m, or 2h")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    return amount * 3600


def _cmd_models() -> int:
    if not os.environ.get("CURSOR_API_KEY"):
        print("Set CURSOR_API_KEY first (see .env.example)", file=sys.stderr)
        return 1
    from cursor_sdk import Cursor

    models = Cursor.models.list()
    print(f"Found {len(models)} models:\n")
    for model in models:
        print(f"  {model.id}")
    return 0


def _cmd_stats(brain: Brain) -> int:
    stats = brain.memory.stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Runs consumed:  {stats['runs_consumed']}")
    print("Tier distribution:")
    for tier, count in stats["tiers"].items():
        print(f"  {tier}: {count}")
    return 0


def _cmd_loop(args: argparse.Namespace) -> int:
    brain = Brain(verbose=args.verbose, allow_agent_runs=args.allow_agent_runs)
    interval = args.every
    runs = 0
    while True:
        runs += 1
        print(f"\n--- Loop run {runs} ---")
        try:
            result = brain.run(args.request, fresh=args.fresh)
            print(result)
        except HybridAgentError as err:
            print(f"Brain error: {err}", file=sys.stderr)
            return 1
        if args.max_runs and runs >= args.max_runs:
            break
        time.sleep(interval)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    brain = Brain(verbose=args.verbose, allow_agent_runs=args.allow_agent_runs)
    try:
        result = brain.run(args.request, fresh=args.fresh)
    except HybridAgentError as err:
        print(f"Brain error: {err}", file=sys.stderr)
        return 1
    print(result)
    return 0


def _build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain", description="HybridAgent brain CLI")
    parser.add_argument("--verbose", action="store_true", help="Print stage boundaries")
    parser.add_argument("--fresh", action="store_true", help="Bypass router cache")
    parser.add_argument(
        "--allow-agent-runs",
        action="store_true",
        help="Enable cursor_agent capability (can edit files)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        _build_base_parser().print_help()
        return 1

    if argv[0] == "models":
        return _cmd_models()

    if argv[0] == "stats":
        return _cmd_stats(Brain())

    if argv[0] == "loop":
        parser = _build_base_parser()
        loop_parser = argparse.ArgumentParser(prog="brain loop", parents=[parser], add_help=True)
        loop_parser.add_argument("command", nargs="?", default="loop", help=argparse.SUPPRESS)
        loop_parser.add_argument("request", help="Request to run repeatedly")
        loop_parser.add_argument("--every", type=_parse_duration, required=True, help="Interval (e.g. 15m)")
        loop_parser.add_argument("--max-runs", type=int, default=0, help="Stop after N runs (0 = forever)")
        args = loop_parser.parse_args(argv)
        return _cmd_loop(args)

    parser = _build_base_parser()
    parser.add_argument("request", nargs=argparse.REMAINDER, help="One-shot request")
    args = parser.parse_args(argv)
    request = " ".join(args.request).strip()
    if not request:
        parser.print_help()
        return 1
    args.request = request
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
