"""Tests for the Executor module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_forge.backtrack import BacktrackEngine
from agent_forge.executor import (
    ExecutionEvent,
    ExecutionResult,
    Executor,
    PlanStep,
    StepResult,
    StepStatus,
)
from agent_forge.models.base import ModelResponse
from agent_forge.monitor import Monitor
from agent_forge.tools.function import FunctionTool
from tests.conftest import MockModel

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


class TestOnStepCompleteCallback:
    """Tests for the on_step_complete progress callback."""

    @pytest.mark.asyncio
    async def test_callback_invoked_for_each_step(self, executor: Executor) -> None:
        """Callback should be called once per completed step."""
        received: list[StepResult] = []

        def on_step(sr: StepResult) -> None:
            received.append(sr)

        steps = [
            PlanStep(id="a", description="Step A"),
            PlanStep(id="b", description="Step B"),
            PlanStep(id="c", description="Step C"),
        ]
        result = await executor.execute_plan(
            steps=steps, invariants=[], on_step_complete=on_step
        )
        assert result.success is True
        assert len(received) == 3
        assert received[0].step_id == "a"
        assert received[0].step_index == 1
        assert received[2].step_index == 3
        assert received[2].steps_total == 3
        assert all(sr.status == StepStatus.COMPLETED for sr in received)

    @pytest.mark.asyncio
    async def test_callback_not_required(self, executor: Executor) -> None:
        """Execution should work fine without a callback."""
        steps = [PlanStep(description="No callback step")]
        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_stream_plan_emits_lifecycle_events(self, executor: Executor) -> None:
        """stream_plan should emit start, step, and completion events."""
        steps = [
            PlanStep(id="a", description="Step A"),
            PlanStep(id="b", description="Step B"),
        ]

        events = [
            event async for event in executor.stream_plan(steps=steps, invariants=[])
        ]

        assert isinstance(events[0], ExecutionEvent)
        assert events[0].event == "plan_started"
        assert [
            event.step_id for event in events if event.event == "step_completed"
        ] == [
            "a",
            "b",
        ]
        assert events[-1].event == "plan_completed"
        assert events[-1].status == StepStatus.COMPLETED
        assert events[-1].terminal is True
        assert events[-1].progress_pct == 100.0
        dumped = events[-1].model_dump()
        assert dumped["terminal"] is True
        assert dumped["progress_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_stream_plan_completed_progress_with_invalidated_step(
        self, executor: Executor
    ) -> None:
        """Terminal success should report complete progress even with skipped steps."""
        steps = [
            PlanStep(id="a", description="Step A"),
            PlanStep(id="skip", description="Skip", status=StepStatus.INVALIDATED),
            PlanStep(id="b", description="Step B"),
        ]

        events = [
            event async for event in executor.stream_plan(steps=steps, invariants=[])
        ]

        assert events[-1].event == "plan_completed"
        assert events[-1].step_index == 2
        assert events[-1].steps_total == 3
        assert events[-1].progress_pct == 100.0

    @pytest.mark.asyncio
    async def test_stream_plan_cancels_when_consumer_stops(
        self, state_manager: StateManager
    ) -> None:
        """Breaking out of the event stream should cancel execution."""
        import asyncio

        async def slow_tool() -> dict[str, str]:
            await asyncio.sleep(10)
            return {"ran": "yes"}

        tool = FunctionTool(fn=slow_tool, name="slow_tool")
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            tools={"slow_tool": tool},
        )

        stream = executor.stream_plan(
            steps=[PlanStep(description="slow", action="slow_tool")],
            invariants=[],
        )
        event = await anext(stream)
        assert event.event == "plan_started"
        assert event.progress_pct == 0.0
        await stream.aclose()

        await asyncio.sleep(0)
        assert state_manager.get("ran") is None

    @pytest.mark.asyncio
    async def test_stream_plan_emits_failure_event(
        self, state_manager: StateManager
    ) -> None:
        """Failures should be exposed as terminal stream events."""

        def fail_tool() -> None:
            raise RuntimeError("boom")

        tool = FunctionTool(fn=fail_tool, name="fail_tool")
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            tools={"fail_tool": tool},
        )

        events = [
            event
            async for event in executor.stream_plan(
                steps=[PlanStep(id="bad", description="bad", action="fail_tool")],
                invariants=[],
            )
        ]

        assert events[-1].event == "plan_failed"
        assert events[-1].status == StepStatus.FAILED
        assert events[-1].step_index == 0
        assert events[-1].steps_total == 1
        assert events[-1].error is not None
        assert "boom" in events[-1].error
        assert events[-1].terminal is True


class TestCostTracking:
    """Tests for execution cost tracking fields."""

    @pytest.mark.asyncio
    async def test_cost_fields_default_to_zero(self, executor: Executor) -> None:
        """Without an LLM, cost fields should be zero."""
        steps = [PlanStep(description="No model step")]
        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.total_tokens == 0
        assert result.total_cost_usd == 0.0
        assert result.model_calls == 0

    @pytest.mark.asyncio
    async def test_cost_fields_populated_with_model(
        self, state_manager: StateManager
    ) -> None:
        """When an LLM is used, cost fields should reflect usage."""
        model = MockModel(responses=["result one", "result two"])
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            model=model,
        )

        steps = [
            PlanStep(description="First LLM step"),
            PlanStep(description="Second LLM step"),
        ]
        result = await executor.execute_plan(steps=steps, invariants=[])
        assert result.success is True
        assert result.model_calls == 2
        # MockModel returns usage={prompt_tokens:10, completion_tokens:20, total_tokens:30}
        assert result.total_tokens == 60
        assert result.total_cost_usd > 0.0

    @pytest.mark.asyncio
    async def test_cost_fields_reset_between_runs(
        self, state_manager: StateManager
    ) -> None:
        """Repeated executions on one executor should not accumulate totals."""
        model = MockModel(responses=["run one", "run two"])
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            model=model,
        )

        first = await executor.execute_plan(
            steps=[PlanStep(description="First LLM step")],
            invariants=[],
        )
        second = await executor.execute_plan(
            steps=[PlanStep(description="Second LLM step")],
            invariants=[],
        )

        assert first.model_calls == 1
        assert first.total_tokens == 30
        assert second.model_calls == 1
        assert second.total_tokens == 30

    @pytest.mark.asyncio
    async def test_checkpoint_counter_resets_between_runs(
        self, state_manager: StateManager
    ) -> None:
        """Checkpoint cadence should not carry over between executions."""
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            checkpoint_interval=2,
        )

        await executor.execute_plan(
            steps=[PlanStep(description="run one step")],
            invariants=[],
        )
        checkpoints_after_first = len(state_manager.checkpoints)

        await executor.execute_plan(
            steps=[PlanStep(description="run two step")],
            invariants=[],
        )

        # Each one-step execution should add only its own plan_start checkpoint.
        assert checkpoints_after_first == 1
        assert len(state_manager.checkpoints) == 2


class TestStepTimeouts:
    """Tests for per-step execution timeouts."""

    @pytest.mark.asyncio
    async def test_async_tool_timeout_fails_recoverably(
        self, state_manager: StateManager
    ) -> None:
        """A slow tool should fail the step without hanging execution."""
        import asyncio

        async def slow_tool() -> dict[str, str]:
            await asyncio.sleep(10)
            return {"ran": "yes"}

        tool = FunctionTool(fn=slow_tool, name="slow_tool")
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            tools={"slow_tool": tool},
            step_timeout_seconds=0.01,
        )
        step = PlanStep(id="slow", description="slow", action="slow_tool")

        result = await executor.execute_plan(steps=[step], invariants=[])

        assert result.success is False
        assert result.recoverable is True
        assert result.steps_completed == 0
        assert "timed out after 0.01s" in result.error
        assert step.status == StepStatus.FAILED
        assert state_manager.get("ran") is None

    @pytest.mark.asyncio
    async def test_model_timeout_fails_before_cost_is_counted(
        self, state_manager: StateManager
    ) -> None:
        """A slow LLM call should be bounded by the configured timeout."""
        import asyncio

        class SlowModel(MockModel):
            async def generate(
                self,
                messages: list[dict[str, str]],
                tools: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> ModelResponse:
                await asyncio.sleep(10)
                return await super().generate(messages, tools=tools, **kwargs)

        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            model=SlowModel(),
            step_timeout_seconds=0.01,
        )

        result = await executor.execute_plan(
            steps=[PlanStep(id="llm-slow", description="slow LLM step")],
            invariants=[],
        )

        assert result.success is False
        assert "Step 'llm-slow' timed out after 0.01s" in result.error
        assert result.model_calls == 0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_stream_plan_emits_timeout_failure(
        self, state_manager: StateManager
    ) -> None:
        """Timeouts should surface as terminal stream failure events."""
        import asyncio

        async def slow_tool() -> dict[str, str]:
            await asyncio.sleep(10)
            return {"ran": "yes"}

        tool = FunctionTool(fn=slow_tool, name="slow_tool")
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)
        executor = Executor(
            state_manager=state_manager,
            monitor=monitor,
            backtrack_engine=backtrack,
            tools={"slow_tool": tool},
            step_timeout_seconds=0.01,
        )

        events = [
            event
            async for event in executor.stream_plan(
                steps=[PlanStep(id="slow", description="slow", action="slow_tool")],
                invariants=[],
            )
        ]

        assert events[-1].event == "plan_failed"
        assert events[-1].status == StepStatus.FAILED
        assert events[-1].error is not None
        assert "Step 'slow' timed out after 0.01s" in events[-1].error
        assert events[-1].terminal is True

    def test_timeout_must_be_positive(self, state_manager: StateManager) -> None:
        """Timeout configuration should reject non-positive values."""
        monitor = Monitor(state_manager=state_manager)
        backtrack = BacktrackEngine(state_manager=state_manager)

        with pytest.raises(ValueError, match="step_timeout_seconds must be positive"):
            Executor(
                state_manager=state_manager,
                monitor=monitor,
                backtrack_engine=backtrack,
                step_timeout_seconds=0,
            )


class TestExecutionResultSerialization:
    """Tests for JSON serialization / deserialization of ExecutionResult."""

    def test_roundtrip_empty_result(self) -> None:
        """An empty result should survive a JSON roundtrip."""
        original = ExecutionResult()
        json_str = original.to_json()
        restored = ExecutionResult.from_json(json_str)
        assert restored == original

    @pytest.mark.asyncio
    async def test_roundtrip_after_execution(self, executor: Executor) -> None:
        """A result from actual execution should roundtrip cleanly."""
        steps = [
            PlanStep(
                id="rt-1",
                description="Roundtrip step",
                expected_state_changes={"key": "value"},
            ),
        ]
        original = await executor.execute_plan(steps=steps, invariants=[])
        json_str = original.to_json()
        restored = ExecutionResult.from_json(json_str)

        assert restored.success == original.success
        assert restored.steps_completed == original.steps_completed
        assert restored.final_state == original.final_state
        assert restored.total_tokens == original.total_tokens
        assert restored.model_calls == original.model_calls
        assert len(restored.verdicts) == len(original.verdicts)
        assert len(restored.plan_trace) == len(original.plan_trace)

    def test_roundtrip_with_cost_data(self) -> None:
        """Cost tracking fields should survive serialization."""
        original = ExecutionResult(
            success=True,
            steps_completed=5,
            steps_total=5,
            total_tokens=1500,
            total_cost_usd=0.0225,
            model_calls=5,
            error="",
        )
        json_str = original.to_json()
        restored = ExecutionResult.from_json(json_str)
        assert restored.total_tokens == 1500
        assert restored.total_cost_usd == pytest.approx(0.0225)
        assert restored.model_calls == 5
