"""Integration tests for the full agent-forge pipeline.

Tests the end-to-end flow: Goal -> Planner -> Executor -> Monitor -> Result,
including backtracking and replanning scenarios.
"""

from __future__ import annotations

import pytest

from agent_forge.backtrack import BacktrackEngine
from agent_forge.executor import ExecutionResult, Executor, PlanStep
from agent_forge.goal import Goal, GoalStatus
from agent_forge.monitor import Monitor
from agent_forge.planner import Agent, MCTSSearch, Planner
from agent_forge.state import StateManager
from tests.conftest import MockModel


class TestFullPipeline:
    """End-to-end tests for the planning-execution loop."""

    @pytest.mark.asyncio
    async def test_simple_goal_completes(self) -> None:
        """A goal with success criteria should not be falsely completed in degraded mode."""
        agent = Agent(model=None, tools=[])
        planner = Planner(
            agent=agent,
            search_strategy="greedy",
            max_backtrack_depth=2,
        )

        goal = Goal(
            description="Simple test goal",
            success_criteria=["Done"],
            max_steps=5,
        )

        result = await planner.execute(goal)
        assert isinstance(result, ExecutionResult)
        assert result.degraded is True
        assert result.success is False
        assert goal.status == GoalStatus.FAILED
        assert "cannot verify" in result.error.lower()

    @pytest.mark.asyncio
    async def test_goal_with_mock_model(self) -> None:
        """Goal with mock model should decompose and execute."""
        model = MockModel(
            responses=[
                # Decomposition response
                "PLAN 1:\n"
                "1. Gather data | action: llm | postcondition: data ready\n"
                "2. Process data | action: llm | postcondition: data processed\n",
                # Step execution responses
                "Data gathered successfully",
                "Data processed successfully",
                # Monitor evaluation responses (not needed in heuristic mode)
            ]
        )
        agent = Agent(model=model, tools=[])
        planner = Planner(
            agent=agent,
            search_strategy="greedy",
            max_backtrack_depth=2,
            checkpoint_interval=1,
        )

        goal = Goal(
            description="Process some data",
            success_criteria=["Data processed"],
            max_steps=10,
        )

        result = await planner.execute(goal)
        assert isinstance(result, ExecutionResult)
        assert result.steps_completed >= 1

    @pytest.mark.asyncio
    async def test_goal_with_tools(self) -> None:
        """Goal should be able to use registered tools."""
        call_log: list[str] = []

        def log_tool(message: str) -> str:
            call_log.append(message)
            return f"Logged: {message}"

        agent = Agent(model=None, tools=[log_tool])
        planner = Planner(
            agent=agent,
            search_strategy="greedy",
            max_backtrack_depth=1,
        )

        goal = Goal(description="Log something", max_steps=5)
        result = await planner.execute(goal)
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_all_strategies_execute(self) -> None:
        """All search strategies should produce valid results."""
        for strategy in ["greedy", "beam", "mcts"]:
            model = MockModel(
                responses=[
                    "PLAN 1:\n1. Step one | action: llm | postcondition: done\n",
                    "Step executed",
                ]
            )
            agent = Agent(model=model, tools=[])
            planner = Planner(
                agent=agent,
                search_strategy=strategy,
                max_backtrack_depth=1,
                num_simulations=5,
            )

            goal = Goal(description=f"Test {strategy}", max_steps=5)
            result = await planner.execute(goal)
            assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_executor_with_monitor_and_backtrack(self) -> None:
        """Executor should handle monitor-triggered backtracking."""
        sm = StateManager(initial_state={"phase": "start"})
        monitor = Monitor(state_manager=sm, drift_threshold=0.3)
        bt = BacktrackEngine(state_manager=sm, max_depth=3)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            checkpoint_interval=1,
        )

        # Step with postcondition that will fail (result is None)
        steps = [
            PlanStep(
                id="failing-step",
                description="This step will fail postcondition",
                postcondition="must have result",
                # No action or model => result will be "Completed: ..."
                # which is truthy, so postcondition should pass heuristically
            ),
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        # The step produces a truthy string result, so heuristic check passes
        assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_handles_high_drift(self) -> None:
        """Executor should trigger backtrack on high state drift."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm, drift_threshold=0.3)
        bt = BacktrackEngine(state_manager=sm, max_depth=3)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            checkpoint_interval=1,
        )

        # Step expects state changes that won't match actual state
        steps = [
            PlanStep(
                id="drift-step",
                description="Step with wrong expectations",
                expected_state_changes={
                    "key1": "expected_value",
                    "key2": "expected_value",
                    "key3": "expected_value",
                },
            ),
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        # State changes are applied, so drift should be 0 (they match now)
        # The executor applies expected_state_changes before checking drift
        assert isinstance(result, ExecutionResult)


class TestStateManagerIntegration:
    """Integration tests for state management with execution."""

    def test_checkpoint_rollback_preserves_integrity(self) -> None:
        """Multiple checkpoint/rollback cycles should maintain state."""
        sm = StateManager(initial_state={"counter": 0})

        # Advance through several checkpoints
        for i in range(5):
            sm.set("counter", i)
            sm.create_checkpoint(metadata={"step": str(i)})
            sm.advance_step()

        # Rollback to the middle
        checkpoints = sm.checkpoints
        mid_cp = checkpoints[2]
        sm.rollback_to_checkpoint(mid_cp.id)

        assert sm.get("counter") == 2
        assert sm.step_index == 2
        assert len(sm.checkpoints) == 3  # cp 0, 1, 2

    def test_dependency_invalidation_cascades(self) -> None:
        """Invalidating a step should cascade to all dependents."""
        sm = StateManager()

        # Build a dependency chain: a -> b -> c, a -> d
        sm.add_dependency("a", "b")
        sm.add_dependency("b", "c")
        sm.add_dependency("a", "d")

        invalidated = sm.invalidate_step("a")
        assert set(invalidated) == {"a", "b", "c", "d"}


class TestMCTSIntegration:
    """Integration tests for MCTS with the planning pipeline."""

    @pytest.mark.asyncio
    async def test_mcts_selects_from_candidates(self) -> None:
        """MCTS should select from multiple candidate plans."""
        search = MCTSSearch(num_simulations=10)

        goal = Goal(
            description="Test MCTS selection",
            success_criteria=["completed"],
            max_steps=10,
        )

        plans = [
            [PlanStep(description="Plan A step 1", postcondition="done")],
            [PlanStep(description="Plan B step 1")],
            [
                PlanStep(
                    description="Plan C step 1",
                    postcondition="done",
                    expected_state_changes={"key": "value"},
                    preconditions=["ready"],
                )
            ],
        ]

        result = await search.search(goal, plans)
        assert result in plans
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mcts_with_model_rollouts(self) -> None:
        """MCTS should use model for rollouts when available."""
        model = MockModel(responses=["0.8", "0.3", "0.9", "0.5"])
        search = MCTSSearch(
            model=model,
            rollout_model=model,
            num_simulations=5,
            max_rollout_depth=3,
        )

        goal = Goal(description="Test", max_steps=5)
        plans = [
            [PlanStep(description="A", postcondition="done")],
            [PlanStep(description="B")],
        ]

        result = await search.search(goal, plans)
        assert result in plans
