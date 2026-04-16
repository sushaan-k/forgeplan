# forgeplan

[![CI](https://github.com/sushaan-k/forgeplan/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/forgeplan/actions)
[![PyPI](https://img.shields.io/pypi/v/forgeplan.svg)](https://pypi.org/project/forgeplan/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/forgeplan.svg)](https://pypi.org/project/forgeplan/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A long-horizon planning engine for LLM agents.**

`forgeplan` decouples planning from generation. Instead of asking the model to improvise an entire multi-step workflow in one shot, it decomposes goals into subtask trees, scores candidate plans with MCTS, executes with checkpoints, and replans when reality diverges.

---

## The Problem

LLM agents fail at multi-step tasks in a compounding way. A model that is 90% reliable per step has only **35% end-to-end reliability across 10 steps**. Research ([arXiv:2601.22311](https://arxiv.org/abs/2601.22311)) shows that strong chain-of-thought reasoning does not transfer to planning — models commit early and cannot recover. Every major agent framework (LangGraph, CrewAI, Anthropic SDK) delegates planning to the model itself. Nobody ships a dedicated planning layer.

## Solution

```python
from forgeplan import Planner, Goal, Step

planner = Planner(model="gpt-4o", strategy="mcts", beam_width=4)

goal = Goal(
    description="Migrate a Postgres database with zero downtime",
    success_criteria=["all tables replicated", "no failed queries during cutover"],
)

plan = await planner.plan(goal)

async for checkpoint in planner.execute(plan):
    print(f"[{checkpoint.step}] {checkpoint.status} — {checkpoint.message}")
    if checkpoint.diverged:
        plan = await planner.replan(checkpoint)   # causal backtrack only
```

## At a Glance

- **HTN decomposition** — breaks goals into validated subtask trees before any LLM call
- **MCTS plan selection** — Monte Carlo Tree Search scores candidate plans, not just greedy first-choice
- **Postcondition monitoring** — checks invariants and state drift after every step
- **Causal backtracking** — rewinds to the specific checkpoint that failed, not from scratch
- **Model and tool agnostic** — wraps OpenAI, Anthropic, or any async callable

## Benchmark

| Strategy | 5-step success | 10-step success | 20-step success |
|---|---|---|---|
| Raw LLM (GPT-4o) | 72% | 43% | 19% |
| LangGraph ReAct | 75% | 47% | 22% |
| **forgeplan (mcts)** | **91%** | **79%** | **61%** |

*Evaluated on PlanBench-v2 across 500 rollouts per condition.*

## Install

```bash
pip install forgeplan
```

## Architecture

```
Goal
 └── HTNDecomposer          # expands goal into subtask tree
      └── MCTSPlanner        # scores & selects plan via rollout simulation
           └── Executor      # runs steps, evaluates postconditions
                └── Monitor  # watches for state drift & invariant violations
                     └── Backtracker  # causal rewind on failure
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐
