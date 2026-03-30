"""GAIA benchmark harness for agent-forge.

GAIA (General AI Assistants) is a benchmark of complex real-world queries
that require multi-step reasoning, tool use, and web navigation.
Current SOTA: ~61% (Writer's Action Agent).

This module provides a harness for running agent-forge against GAIA tasks
and measuring planning effectiveness vs. vanilla agents.

Usage:
    python -m tests.benchmarks.gaia --tasks 10 --strategy mcts

Note:
    Requires API keys and the GAIA dataset. This module defines the
    harness structure; actual benchmark runs need environment setup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agent_forge import Agent, Goal, Planner

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark task.

    Attributes:
        task_id: Identifier of the GAIA task.
        success: Whether the task was completed successfully.
        steps_completed: Number of steps executed.
        backtracks: Number of backtracking events triggered.
        wall_time_seconds: Total wall-clock time.
        token_cost: Estimated token usage.
        error: Error message if the task failed.
    """

    task_id: str
    success: bool
    steps_completed: int
    backtracks: int
    wall_time_seconds: float
    token_cost: int = 0
    error: str = ""


@dataclass
class BenchmarkSuite:
    """Collection of GAIA benchmark results.

    Attributes:
        results: Individual task results.
        strategy: Search strategy used.
        model: Model used for the benchmark.
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    strategy: str = "mcts"
    model: str = "claude-sonnet-4-6"

    @property
    def success_rate(self) -> float:
        """Fraction of tasks completed successfully."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_steps(self) -> float:
        """Average steps across all tasks."""
        if not self.results:
            return 0.0
        return sum(r.steps_completed for r in self.results) / len(self.results)

    @property
    def avg_backtracks(self) -> float:
        """Average backtracks per task."""
        if not self.results:
            return 0.0
        return sum(r.backtracks for r in self.results) / len(self.results)

    def summary(self) -> dict[str, Any]:
        """Generate a summary of benchmark results."""
        return {
            "total_tasks": len(self.results),
            "success_rate": round(self.success_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_backtracks": round(self.avg_backtracks, 2),
            "total_time": round(sum(r.wall_time_seconds for r in self.results), 2),
            "strategy": self.strategy,
            "model": self.model,
        }


# Sample GAIA-style tasks for testing the harness
SAMPLE_TASKS: list[dict[str, Any]] = [
    {
        "id": "gaia-001",
        "description": (
            "Find the current CEO of NVIDIA, their total compensation "
            "from the most recent proxy statement, and compare it to "
            "AMD's CEO compensation."
        ),
        "criteria": [
            "NVIDIA CEO name identified",
            "NVIDIA CEO compensation figure cited with source",
            "AMD CEO compensation figure cited with source",
            "Comparison provided",
        ],
    },
    {
        "id": "gaia-002",
        "description": (
            "Calculate the population-weighted average temperature "
            "in January for the 5 largest US cities. Use census data "
            "for populations and NOAA data for temperatures."
        ),
        "criteria": [
            "5 cities identified with populations",
            "January average temperatures found for each",
            "Population-weighted average computed correctly",
            "Data sources cited",
        ],
    },
    {
        "id": "gaia-003",
        "description": (
            "Find all Python packages on PyPI that have 'agent' in "
            "the name, were updated in the last 30 days, and have "
            "more than 1000 GitHub stars. List them with star counts."
        ),
        "criteria": [
            "PyPI search performed",
            "Packages filtered by recency",
            "GitHub stars checked for each",
            "Final list with star counts provided",
        ],
    },
]


async def run_task(
    task: dict[str, Any],
    strategy: str = "mcts",
    model_name: str = "claude-sonnet-4-6",
) -> BenchmarkResult:
    """Run a single GAIA-style benchmark task.

    Args:
        task: Task definition with id, description, and criteria.
        strategy: Search strategy to use.
        model_name: LLM model identifier.

    Returns:
        BenchmarkResult for the task.
    """
    goal = Goal(
        description=task["description"],
        success_criteria=task["criteria"],
        max_steps=50,
        invariants=["Never fabricate data"],
    )

    agent = Agent(model=model_name, tools=[])
    planner = Planner(
        agent=agent,
        search_strategy=strategy,
        max_backtrack_depth=5,
        checkpoint_interval=3,
    )

    start = time.monotonic()
    result = await planner.execute(goal)
    elapsed = time.monotonic() - start

    return BenchmarkResult(
        task_id=task["id"],
        success=result.success,
        steps_completed=result.steps_completed,
        backtracks=sum(1 for v in result.verdicts if v.should_backtrack),
        wall_time_seconds=elapsed,
        error=result.error,
    )


async def run_suite(
    tasks: list[dict[str, Any]] | None = None,
    strategy: str = "mcts",
    model_name: str = "claude-sonnet-4-6",
) -> BenchmarkSuite:
    """Run the full GAIA benchmark suite.

    Args:
        tasks: Task definitions. Defaults to SAMPLE_TASKS.
        strategy: Search strategy.
        model_name: LLM model identifier.

    Returns:
        BenchmarkSuite with all results.
    """
    tasks = tasks or SAMPLE_TASKS
    suite = BenchmarkSuite(strategy=strategy, model=model_name)

    for task in tasks:
        logger.info("Running task %s", task["id"])
        result = await run_task(task, strategy, model_name)
        suite.results.append(result)
        logger.info(
            "Task %s: success=%s steps=%d time=%.1fs",
            result.task_id,
            result.success,
            result.steps_completed,
            result.wall_time_seconds,
        )

    return suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    suite = asyncio.run(run_suite())
    print("\nBenchmark Summary:")
    for key, value in suite.summary().items():
        print(f"  {key}: {value}")
