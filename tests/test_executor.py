"""Tests for the Executor module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_forge.backtrack import BacktrackEngine
from agent_forge.executor import (
    ExecutionResult,
    Executor,
    PlanStep,
    StepStatus,
)
from agent_forge.monitor import Monitor
from agent_forge.tools.function import FunctionTool

if TYPE_CHECKING:
    from agent_forge.state import StateManager


class TestPlanStep:
    """Tests for PlanStep data model."""

    def test_create_step_defaults(self) -> None:
        """PlanStep should have sensible defaults."""
        step = PlanStep(description="Test step")
        assert step.description == "Test step"
        assert step.status == StepStatus.PENDING
        assert step.result is None
        assert step.subtasks == []
        assert step.depends_on == []
        assert len(step.id) == 12

    def test_create_step_full(self) -> None:
        """PlanStep should accept all fields."""
        step = PlanStep(
            description="Full step",
            action="tool_name",
            action_args={"key": "value"},
            preconditions=["ready"],
            postcondition="done",
            expected_state_changes={"completed": True},
            depends_on=["prev-step"],
        )
        assert step.action == "tool_name"
        assert step.action_args == {"key": "value"}
        assert step.postcondition == "done"


class TestExecutor:
    """Tests for plan execution."""

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self, executor: Executor) -> None:
        """Empty plan should succeed immediately."""
        result = await executor.execute_plan(steps=[], invariants=[])
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.steps_completed == 0

    @pytest.mark.asyncio
    async def test_execute_single_step(self, executor: Executor) -> None:
        """Single step plan should execute and succeed."""
        steps = [
            PlanStep(
                description="Simple step",
                postcondition="done",
            )
        ]
        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert result.steps_completed == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_steps(
        self, executor: Executor, sample_steps: list[PlanStep]
    ) -> None:
        """Multi-step plan should execute all steps."""
        result = await executor.execute_plan(steps=sample_steps, invariants=[])
        assert result.success is True
        assert result.steps_completed == 3

    @pytest.mark.asyncio
    async def test_execute_with_tool(self, state_manager: StateManager) -> None:
        """Should execute steps via registered tools."""

        def compute(expression: str) -> str:
            return str(eval(expression))

        tool = FunctionTool(fn=compute, name="compute")
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            tools={"compute": tool},
        )

        steps = [
            PlanStep(
                description="Calculate 2+2",
                action="compute",
                action_args={"expression": "2+2"},
                postcondition="result computed",
            )
        ]

        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert steps[0].result == "4"

    @pytest.mark.asyncio
    async def test_execute_checkpoints_created(
        self, state_manager: StateManager
    ) -> None:
        """Executor should create checkpoints at configured intervals."""
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            checkpoint_interval=2,
        )

        steps = [PlanStep(id=f"s{i}", description=f"Step {i}") for i in range(4)]

        await executor.execute_plan(steps=steps, invariants=[])
        # 1 initial + at least 2 interval checkpoints
        assert len(state_manager.checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_execute_state_updates(
        self, executor: Executor, state_manager: StateManager
    ) -> None:
        """Executor should apply state changes from steps."""
        steps = [
            PlanStep(
                description="Update state",
                expected_state_changes={"phase": "done"},
            )
        ]
        await executor.execute_plan(steps=steps, invariants=[])
        assert state_manager.get("phase") == "done"

    def test_flatten_steps(self) -> None:
        """Should flatten nested subtasks into ordered list."""
        child = PlanStep(id="child", description="Child step")
        parent = PlanStep(id="parent", description="Parent step", subtasks=[child])
        flat = Executor._flatten_steps([parent])
        ids = [s.id for s in flat]
        assert ids.index("child") < ids.index("parent")

    def test_count_steps(self) -> None:
        """Should count total steps including nested subtasks."""
        child = PlanStep(description="Child")
        parent = PlanStep(description="Parent", subtasks=[child])
        assert Executor._count_steps([parent]) == 2
