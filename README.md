# HybridAgent

A minimal two-stage agent inspired by Anthropic's advisor/worker sub-agent pattern: escalate to a strong advisor, delegate to cheap workers. All model calls go through the **Cursor Python SDK** and bill to your Cursor monthly subscription.

## Setup

Requires Python 3.10+.

```bash
cd ~/Projects/hybrid-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set your Cursor API key (from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations)):

```bash
export CURSOR_API_KEY=cursor_your_key_here
```

Usage draws from the same request pools as the Cursor IDE and appears in your usage dashboard under the SDK tag.

## Brain CLI (recommended entry point)

Everything routes through the brain. **Composer 2.5 is always the first model to touch a request** (configurable via `router_model` or `BRAIN_ROUTER_MODEL`).

```bash
# One-shot
brain "Write a launch plan for a new AI coding tool"
brain --verbose "Create a marketing strategy for my SaaS product"
brain --fresh "Run security review"          # bypass cache

# Recurring loop
brain loop "Check open PRs and summarize blockers" --every 15m --max-runs 20

# Utilities
brain models    # list model slugs on your subscription
brain stats     # tier distribution + runs consumed (watch for over-escalation)
```

### Routing tiers (cheapest first)

| Tier | Runs | When |
|------|------|------|
| `direct` | 1 | Simple questions, greetings, one-shot answers |
| `single` | 2 | One clear action mapped to a capability |
| `pipeline` | 4+N | Router + advisor + orchestrator + N workers + synthesis |

Repeat requests can cost **0 runs** via router cache (1h TTL, expired entries auto-pruned). Use `--fresh` to bypass.

### Guardrails & failure handling

- Router failures (bad JSON, exceptions) escalate to `pipeline` instead of crashing; JSON wrapped in prose or fences is tolerated.
- A `direct` route with no usable answer downgrades to a `single` llm call; an unknown `single` capability falls back to `llm`.
- Orchestrator output is validated: max **5 tasks**, max **2 `cursor_agent` tasks**, tasks must be JSON objects with `id`/`title`/`prompt`.
- Worker/executor failures are wrapped in `HybridAgentError` with the failing task id, so the CLI always exits cleanly.
- Cache and run-log corruption is tolerated (bad entries skipped); cache writes are atomic.
- `brain stats` reads the local run log only — works without `cursor-sdk` or an API key.
- `brain loop` rejects zero/negative intervals and negative `--max-runs`.

### Enable real file edits

`cursor_agent` capability runs a full Cursor SDK agent (`mode=agent`). Opt in explicitly:

```bash
brain --allow-agent-runs "Refactor auth middleware and add tests"
```

## Python API

```python
from brain import Brain

brain = Brain(
    router_model="composer-2.5",      # first model to touch every request
    advisor_model="glm-5.2-max",      # escalate up for complex work
    worker_model="composer-2.5",      # delegate down for workers
    verbose=True,
)
result = brain.run("Create a marketing strategy for my SaaS product")
```

Legacy pipeline (no router):

```python
from hybrid_agent import HybridAgent

agent = HybridAgent(verbose=True)
result = agent.process("Write a launch plan for a new AI coding tool")
```

## Capabilities

| Capability | Description |
|------------|-------------|
| `llm` | Text/analysis via model worker (default) |
| `cursor_agent` | Full Cursor SDK agent run against repo (requires `allow_agent_runs=True`) |

## How the pipeline works

```
Request → Router (composer-2.5) → direct | single | pipeline
  pipeline: Advisor → Orchestrator (JSON tasks) → Workers → Synthesis
```

Runs default to **plan mode** (read-only). Pass `mode="agent"` to `HybridAgent` or use `--allow-agent-runs` on the brain for file edits.

## Tests

Unit tests use mocks — no live API calls, and they pass with or without `cursor-sdk` installed:

```bash
pytest
```

## Project layout

```
hybrid-agent/
├── brain.py            # Brain router + entry point
├── hybrid_agent.py     # Advisor/worker pipeline engine
├── capabilities.py   # Capability registry
├── executors.py      # llm + cursor_agent executors
├── router.py         # Fast-path classifier
├── memory.py         # Run log + cache
├── cli.py            # brain CLI
├── callers/
│   └── cursor_sdk.py
├── examples/
└── tests/
```
