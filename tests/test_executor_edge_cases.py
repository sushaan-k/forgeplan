"""Edge case tests for the Executor module.

Covers tool failures, async tool errors, invalidated steps,
deeply nested subtask trees, dependency registration during
execution, and checkpoint interval behavior.
"""

from __future__ import annotations

import pytest

from agent_forge.backtrack import BacktrackEngine
from agent_forge.executor import Executor, PlanStep, StepStatus
from agent_forge.monitor import Monitor
from agent_forge.state import StateManager
from agent_forge.tools.function import FunctionTool
from tests.conftest import MockModel


class TestPlanStepEdgeCases:
    """Edge cases for PlanStep model."""

    def test_step_with_empty_description(self) -> None:
        """PlanStep with empty description should be valid."""
        step = PlanStep(description="")
        assert step.description == ""
        assert step.status == StepStatus.PENDING

    def test_step_with_deep_subtasks(self) -> None:
        """PlanStep should handle deeply nested subtasks."""
        inner = PlanStep(description="innermost")
        mid = PlanStep(description="middle", subtasks=[inner])
        outer = PlanStep(description="outer", subtasks=[mid])

        flat = Executor._flatten_steps([outer])
        descriptions = [s.description for s in flat]
        assert descriptions == ["innermost", "middle", "outer"]

    def test_step_with_multiple_subtask_branches(self) -> None:
        """Flatten should handle multiple subtask branches."""
        leaf1 = PlanStep(id="leaf1", description="leaf 1")
        leaf2 = PlanStep(id="leaf2", description="leaf 2")
        branch1 = PlanStep(id="b1", description="branch 1", subtasks=[leaf1])
        branch2 = PlanStep(id="b2", description="branch 2", subtasks=[leaf2])
        root = PlanStep(id="root", description="root", subtasks=[branch1, branch2])

        flat = Executor._flatten_steps([root])
        ids = [s.id for s in flat]
        # Subtasks come before their parents
        assert ids.index("leaf1") < ids.index("b1")
        assert ids.index("leaf2") < ids.index("b2")
        assert ids.index("b1") < ids.index("root")
        assert ids.index("b2") < ids.index("root")

    def test_count_steps_empty(self) -> None:
        """Counting steps on empty list should return 0."""
        assert Executor._count_steps([]) == 0

    def test_count_steps_deeply_nested(self) -> None:
        """Should count all steps in a deep tree."""
        inner = PlanStep(description="inner")
        mid = PlanStep(description="mid", subtasks=[inner])
        outer = PlanStep(description="outer", subtasks=[mid])
        assert Executor._count_steps([outer]) == 3

    def test_step_unique_ids(self) -> None:
        """Auto-generated step IDs should be unique."""
        steps = [PlanStep(description=f"step {i}") for i in range(50)]
        ids = {s.id for s in steps}
        assert len(ids) == 50


class TestExecutorToolFailures:
    """Tests for tool execution failures in the executor."""

    @pytest.mark.asyncio
    async def test_tool_failure_produces_execution_error(self) -> None:
        """When a tool raises, executor should catch and return error."""

        def failing_tool() -> None:
            raise RuntimeError("disk full")

        tool = FunctionTool(fn=failing_tool, name="fail_tool")
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            tools={"fail_tool": tool},
        )

        steps = [
            PlanStep(
                id="s1",
                description="Call failing tool",
                action="fail_tool",
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is False
        assert "disk full" in result.error

    @pytest.mark.asyncio
    async def test_async_tool_failure(self) -> None:
        """Async tool failure should be caught as ExecutionError."""

        async def async_fail(x: int) -> int:
            raise ValueError(f"invalid x={x}")

        tool = FunctionTool(fn=async_fail, name="async_fail")
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            tools={"async_fail": tool},
        )

        steps = [
            PlanStep(
                id="s1",
                description="Async failing tool",
                action="async_fail",
                action_args={"x": -1},
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is False

    @pytest.mark.asyncio
    async def test_first_step_fails_stops_execution(self) -> None:
        """If the first step fails, remaining steps should not execute."""

        def fail() -> None:
            raise RuntimeError("early failure")

        tool = FunctionTool(fn=fail, name="fail")
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            tools={"fail": tool},
        )

        steps = [
            PlanStep(id="s1", description="Fail", action="fail"),
            PlanStep(id="s2", description="Should not run"),
            PlanStep(id="s3", description="Should not run"),
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is False
        assert result.steps_completed == 0
        assert steps[1].status == StepStatus.PENDING
        assert steps[2].status == StepStatus.PENDING


class TestExecutorWithInvalidatedSteps:
    """Tests for handling invalidated steps during execution."""

    @pytest.mark.asyncio
    async def test_invalidated_steps_skipped(self) -> None:
        """Steps marked as INVALIDATED should be skipped."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
        )

        steps = [
            PlanStep(id="s1", description="Normal step"),
            PlanStep(
                id="s2",
                description="Invalidated step",
                status=StepStatus.INVALIDATED,
            ),
            PlanStep(id="s3", description="Another normal step"),
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert result.steps_completed == 2
        assert steps[1].status == StepStatus.INVALIDATED


class TestExecutorCheckpointing:
    """Tests for checkpoint creation during execution."""

    @pytest.mark.asyncio
    async def test_checkpoint_interval_1(self) -> None:
        """With interval=1, should checkpoint after every step."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            checkpoint_interval=1,
        )

        steps = [PlanStep(id=f"s{i}", description=f"Step {i}") for i in range(5)]
        await executor.execute_plan(steps=steps, invariants=[])

        # 1 initial + 5 per-step checkpoints
        assert len(sm.checkpoints) >= 6

    @pytest.mark.asyncio
    async def test_checkpoint_interval_larger_than_steps(self) -> None:
        """With interval larger than step count, only initial checkpoint created."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            checkpoint_interval=100,
        )

        steps = [PlanStep(id="s1", description="Single step")]
        await executor.execute_plan(steps=steps, invariants=[])

        # Only the initial plan_start checkpoint
        assert len(sm.checkpoints) == 1


class TestExecutorWithModel:
    """Tests for executor using LLM model for step execution."""

    @pytest.mark.asyncio
    async def test_execute_with_model(self) -> None:
        """Steps without matching tools should use the model."""
        model = MockModel(responses=["Model generated result"])
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            model=model,
        )

        steps = [
            PlanStep(
                id="s1",
                description="Generate something",
                action="llm",
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert steps[0].result == "Model generated result"

    @pytest.mark.asyncio
    async def test_execute_no_model_no_tool_returns_default(self) -> None:
        """With no model and no matching tool, should return default string."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            model=None,
            tools={},
        )

        steps = [
            PlanStep(
                id="s1",
                description="Do something",
                action="unknown_tool",
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert "Completed:" in steps[0].result


class TestExecutorStateUpdates:
    """Tests for state update behavior during execution."""

    @pytest.mark.asyncio
    async def test_state_changes_applied_sequentially(self) -> None:
        """State changes from earlier steps should be visible to later checks."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
        )

        steps = [
            PlanStep(
                id="s1",
                description="Set phase",
                expected_state_changes={"phase": "step1_done"},
            ),
            PlanStep(
                id="s2",
                description="Set result",
                expected_state_changes={"result": "step2_done"},
            ),
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert sm.get("phase") == "step1_done"
        assert sm.get("result") == "step2_done"

    @pytest.mark.asyncio
    async def test_step_index_advances_correctly(self) -> None:
        """Step index should advance once per completed step."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
        )

        steps = [PlanStep(id=f"s{i}", description=f"Step {i}") for i in range(3)]
        await executor.execute_plan(steps=steps, invariants=[])
        assert sm.step_index == 3


class TestExecutorDependencyRegistration:
    """Tests for dependency graph updates during execution."""

    @pytest.mark.asyncio
    async def test_dependencies_registered_during_execution(self) -> None:
        """Executor should register dependencies in the state manager."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
        )

        steps = [
            PlanStep(id="s1", description="Step 1"),
            PlanStep(id="s2", description="Step 2", depends_on=["s1"]),
            PlanStep(id="s3", description="Step 3", depends_on=["s2"]),
        ]

        await executor.execute_plan(steps=steps, invariants=[])
        graph = sm.dependency_graph
        assert graph.has_edge("s1", "s2")
        assert graph.has_edge("s2", "s3")


class TestExecutorDependencyOrdering:
    """Tests for dependency-aware scheduling."""

    @pytest.mark.asyncio
    async def test_out_of_order_steps_run_in_dependency_order(self) -> None:
        """A dependent step should wait until its prerequisite completes."""
        call_order: list[str] = []

        def first_step() -> dict[str, str]:
            call_order.append("first")
            return {"phase": "first"}

        def second_step() -> dict[str, str]:
            call_order.append("second")
            return {"phase": "second"}

        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            tools={
                "first": FunctionTool(fn=first_step, name="first"),
                "second": FunctionTool(fn=second_step, name="second"),
            },
        )

        steps = [
            PlanStep(
                id="step-2",
                description="Second",
                action="second",
                depends_on=["step-1"],
            ),
            PlanStep(
                id="step-1",
                description="First",
                action="first",
            ),
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert call_order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_missing_dependency_fails_fast(self) -> None:
        """A step with an unknown dependency should not execute."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
        )

        steps = [
            PlanStep(
                id="step-1",
                description="Blocked",
                depends_on=["missing-step"],
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is False
        assert "missing dependencies" in result.error


class TestExecutorDriftDetection:
    """Tests for state drift detection behavior."""

    @pytest.mark.asyncio
    async def test_drift_detects_unexpected_state_changes(self) -> None:
        """If actual state differs from the plan, the monitor should backtrack."""

        def actual_state() -> dict[str, str]:
            return {"phase": "unexpected"}

        sm = StateManager(initial_state={"phase": "start"})
        monitor = Monitor(state_manager=sm, drift_threshold=0.1)
        bt = BacktrackEngine(state_manager=sm)
        executor = Executor(
            state_manager=sm,
            monitor=monitor,
            backtrack_engine=bt,
            tools={"stateful": FunctionTool(fn=actual_state, name="stateful")},
        )

        steps = [
            PlanStep(
                id="step-1",
                description="Produce state",
                action="stateful",
                expected_state_changes={"phase": "expected"},
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is False
        assert "Backtracked" in result.error or "Cannot backtrack" in result.error
