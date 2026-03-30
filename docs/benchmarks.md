# Benchmarks

## Overview

agent-forge is designed to be quantitatively evaluated against established agent benchmarks. The benchmarking suite measures planning effectiveness, backtracking frequency, step efficiency, and token cost overhead compared to vanilla (unplanned) agents.

## Benchmark Suites

### GAIA (General AI Assistants)

Complex real-world queries requiring multi-step reasoning, tool use, and web navigation.

- **Current SOTA**: ~61% (Writer's Action Agent)
- **Target**: Demonstrate improvement on tasks exceeding 10 steps
- **Location**: `tests/benchmarks/gaia.py`

**Metrics tracked:**
- Success rate (overall and by task complexity)
- Steps to completion
- Backtrack frequency
- Wall-clock time
- Token cost

### Long-Horizon Suite

Custom tasks specifically designed to stress-test planning at scale. Targets the "early myopic commitment" failure mode described in [arXiv:2601.22311](https://arxiv.org/abs/2601.22311).

- **Task categories**: Linear chains, branching tasks, recovery scenarios
- **Step ranges**: 10, 30, 50+ steps
- **Location**: `tests/benchmarks/long_horizon.py`

**Metrics tracked:**
- Success rate bucketed by horizon length
- Backtrack frequency and depth
- Recovery rate (tasks saved by backtracking)
- Planning overhead vs. vanilla execution

## Running Benchmarks

```bash
# GAIA benchmark (requires API keys)
python -m tests.benchmarks.gaia

# Long-horizon benchmark
python -m tests.benchmarks.long_horizon
```

## Expected Results

### Hypothesis

Based on the research findings:
- Vanilla agents degrade rapidly beyond ~15 steps
- agent-forge's backtracking should recover from early myopic commitments
- MCTS plan selection should outperform greedy on branching tasks
- Checkpoint overhead should be <10% of total step time

### Metrics Template

| Metric | Vanilla | agent-forge (greedy) | agent-forge (MCTS) |
|---|---|---|---|
| Success rate (10 steps) | ~80% | ~85% | ~88% |
| Success rate (30 steps) | ~40% | ~65% | ~72% |
| Success rate (50 steps) | ~15% | ~50% | ~58% |
| Avg backtracks per task | N/A | 1.2 | 1.8 |
| Token overhead | 1x | 1.1x | 1.4x |

*Values are hypothetical targets. Actual results depend on the model, task complexity, and tool availability.*

## Comparing Strategies

To compare search strategies on the same tasks:

```python
import asyncio
from tests.benchmarks.long_horizon import run_suite

async def compare():
    for strategy in ["greedy", "beam", "mcts"]:
        suite = await run_suite(strategy=strategy)
        print(f"\n{strategy}: {suite.summary()}")

asyncio.run(compare())
```

## Adding Custom Benchmarks

Create a task definition following this format:

```python
task = {
    "id": "custom-001",
    "description": "Your task description",
    "criteria": ["Success criterion 1", "Success criterion 2"],
    "invariants": ["Safety constraint"],
    "target_steps": 20,
    "type": "linear_chain",
}
```

Pass it to `run_task()` or `run_suite()` from either benchmark module.
