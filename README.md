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

## Usage

```python
from hybrid_agent import HybridAgent

# Default — all stages use composer-2.5
agent = HybridAgent(verbose=True)
result = agent.process("Write a launch plan for a new AI coding tool")

# Cheap preset — GLM planner + Composer workers
cheap_agent = HybridAgent(
    advisor_model="glm-5.2",
    orchestrator_model="glm-5.2",
    worker_model="composer-2.5",
    verbose=True,
)
result = cheap_agent.process("Create a marketing strategy for my SaaS product")
```

Run the included examples:

```bash
python examples/default_agent.py
python examples/cheap_agent.py
```

## Check available models

Model slugs vary by account. List what's available on your subscription:

```bash
python examples/list_models.py
```

If `glm-5.2` fails, try `glm-5.2-max` or another slug from the list.

## How it works

```
User request → Advisor → Orchestrator (JSON tasks) → Workers → Synthesis → Final answer
```

| Stage | Default model | Role |
|-------|---------------|------|
| Advisor | `composer-2.5` | Strategic planning |
| Orchestrator | `composer-2.5` | Break work into 1–5 subtasks |
| Worker | `composer-2.5` | Execute each subtask |
| Synthesis | orchestrator model | Merge outputs into final answer |

Runs default to **plan mode** (read-only). Pass `mode="agent"` to `HybridAgent` if workers should modify files.

Each `process()` call makes 3 + N Cursor SDK runs (N = number of tasks, capped at 5).

## Tests

Unit tests use a mock `model_caller` — no live API calls:

```bash
pytest
```

Cursor caller tests mock `Agent.prompt` to verify error mapping without billing your subscription.

## Project layout

```
hybrid-agent/
├── hybrid_agent.py       # Core pipeline
├── callers/
│   └── cursor_sdk.py   # Default Cursor SDK caller
├── examples/
├── tests/
└── pyproject.toml
```
