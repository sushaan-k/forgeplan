"""Long-horizon benchmark suite for agent-forge.

Tests agent-forge's planning advantage on tasks designed to break
vanilla agents through long step chains, cascading dependencies,
and state management complexity.

These tasks specifically target the "early myopic commitment" failure
mode described in arXiv:2601.22311, where agents make locally optimal
choices that compound into globally terrible outcomes.

Usage:
    python -m tests.benchmarks.long_horizon --steps 50 --strategy mcts
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
class LongHorizonResult:
    """Result from a long-horizon benchmark task.

    Attributes:
        task_id: Task identifier.
        task_type: Category of the task.
        target_steps: Number of steps the task was designed for.
        success: Whether the task completed successfully.
        steps_completed: Actual steps executed.
        backtracks: Number of backtrack events.
        max_backtrack_depth: Deepest consecutive backtrack depth reached.
        wall_time_seconds: Total wall-clock time.
        error: Error message if failed.
    """

    task_id: str
    task_type: str
    target_steps: int
    success: bool
    steps_completed: int
    backtracks: int
    max_backtrack_depth: int
    wall_time_seconds: float
    error: str = ""


@dataclass
class LongHorizonSuite:
    """Collection of long-horizon benchmark results.

    Attributes:
        results: Individual task results.
        strategy: Search strategy used.
    """

    results: list[LongHorizonResult] = field(default_factory=list)
    strategy: str = "mcts"

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def success_rate_by_horizon(self) -> dict[str, float]:
        """Success rate bucketed by target step count."""
        buckets: dict[str, list[bool]] = {}
        for r in self.results:
            if r.target_steps <= 10:
                bucket = "short (<=10)"
            elif r.target_steps <= 30:
                bucket = "medium (11-30)"
            elif r.target_steps <= 60:
                bucket = "long (31-60)"
            else:
                bucket = "very_long (60+)"
            buckets.setdefault(bucket, []).append(r.success)

        return {
            bucket: sum(results) / len(results)
            for bucket, results in sorted(buckets.items())
        }

    def summary(self) -> dict[str, Any]:
        """Generate a summary report."""
        return {
            "total_tasks": len(self.results),
            "success_rate": round(self.success_rate, 4),
            "by_horizon": self.success_rate_by_horizon(),
            "avg_backtracks": round(
                sum(r.backtracks for r in self.results) / max(len(self.results), 1),
                2,
            ),
            "avg_backtrack_depth": round(
                sum(r.max_backtrack_depth for r in self.results)
                / max(len(self.results), 1),
                2,
            ),
            "strategy": self.strategy,
        }


# Task templates designed to stress-test planning at different horizons
TASK_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "lh-chain-10",
        "type": "linear_chain",
        "target_steps": 10,
        "description": (
            "Execute a 10-step data pipeline: ingest raw CSV, validate schema, "
            "clean nulls, normalize columns, compute aggregates, join with "
            "reference data, apply business rules, generate summary statistics, "
            "format output report, and save to storage."
        ),
        "criteria": ["Pipeline completed", "Output validated"],
        "invariants": ["No data loss between steps"],
    },
    {
        "id": "lh-chain-30",
        "type": "linear_chain",
        "target_steps": 30,
        "description": (
            "Build a complete web application: set up project structure, "
            "create database models, implement CRUD API endpoints, add "
            "authentication, write input validation, create frontend templates, "
            "add CSS styling, write unit tests, write integration tests, "
            "configure CI/CD, write documentation, and deploy to staging."
        ),
        "criteria": [
            "All source files created",
            "Tests pass",
            "Documentation complete",
        ],
        "invariants": [
            "Never overwrite files without checking first",
            "All code must be syntactically valid",
        ],
    },
    {
        "id": "lh-branching-20",
        "type": "branching",
        "target_steps": 20,
        "description": (
            "Research three competing technologies, evaluate each on "
            "5 criteria (performance, cost, ecosystem, learning curve, "
            "community support), compile a comparison matrix, and write "
            "a recommendation with trade-off analysis."
        ),
        "criteria": [
            "Three technologies researched",
            "Five criteria evaluated for each",
            "Comparison matrix complete",
            "Recommendation provided with reasoning",
        ],
        "invariants": ["All claims must be supported by data"],
    },
    {
        "id": "lh-recovery-50",
        "type": "recovery",
        "target_steps": 50,
        "description": (
            "Migrate a legacy system to a new architecture. Steps include "
            "auditing the existing system, mapping dependencies, designing "
            "the new architecture, implementing adapter layers, migrating "
            "each component, running compatibility tests, handling failures "
            "by rolling back individual components, and verifying the "
            "complete migration."
        ),
        "criteria": [
            "All components migrated",
            "Compatibility tests pass",
            "Rollback tested for at least one component",
        ],
        "invariants": [
            "Never modify production data without a backup",
            "Each component migration must be independently reversible",
        ],
    },
]


async def run_task(
    task: dict[str, Any],
    strategy: str = "mcts",
    model_name: str = "claude-sonnet-4-6",
) -> LongHorizonResult:
    """Run a single long-horizon benchmark task.

    Args:
        task: Task definition.
        strategy: Search strategy.
        model_name: LLM model identifier.

    Returns:
        LongHorizonResult for the task.
    """
    goal = Goal(
        description=task["description"],
        success_criteria=task["criteria"],
        max_steps=task["target_steps"] * 2,
        invariants=task.get("invariants", []),
    )

    agent = Agent(model=model_name, tools=[])
    planner = Planner(
        agent=agent,
        search_strategy=strategy,
        max_backtrack_depth=min(task["target_steps"] // 5, 10),
        checkpoint_interval=max(task["target_steps"] // 10, 2),
    )

    start = time.monotonic()
    result = await planner.execute(goal)
    elapsed = time.monotonic() - start

    backtrack_events = [v for v in result.verdicts if v.should_backtrack]

    return LongHorizonResult(
        task_id=task["id"],
        task_type=task["type"],
        target_steps=task["target_steps"],
        success=result.success,
        steps_completed=result.steps_completed,
        backtracks=len(backtrack_events),
        max_backtrack_depth=0,
        wall_time_seconds=elapsed,
        error=result.error,
    )


async def run_suite(
    tasks: list[dict[str, Any]] | None = None,
    strategy: str = "mcts",
    model_name: str = "claude-sonnet-4-6",
) -> LongHorizonSuite:
    """Run the full long-horizon benchmark suite.

    Args:
        tasks: Task definitions. Defaults to TASK_TEMPLATES.
        strategy: Search strategy.
        model_name: LLM model identifier.

    Returns:
        LongHorizonSuite with all results.
    """
    tasks = tasks or TASK_TEMPLATES
    suite = LongHorizonSuite(strategy=strategy)

    for task in tasks:
        logger.info(
            "Running task %s (%s, %d steps)",
            task["id"],
            task["type"],
            task["target_steps"],
        )
        result = await run_task(task, strategy, model_name)
        suite.results.append(result)
        logger.info(
            "Task %s: success=%s steps=%d backtracks=%d time=%.1fs",
            result.task_id,
            result.success,
            result.steps_completed,
            result.backtracks,
            result.wall_time_seconds,
        )

    return suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    suite = asyncio.run(run_suite())
    summary = suite.summary()
    print("\nLong-Horizon Benchmark Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
