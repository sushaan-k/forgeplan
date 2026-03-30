"""Step-by-step plan executor.

The Executor takes a plan (a tree of PlanStep nodes) and runs each step
sequentially, managing tool calls, state updates, and monitoring checks.
It coordinates with the Monitor and BacktrackEngine to handle failures.
"""

from __future__ import annotations

import logging
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agent_forge.backtrack import BacktrackEngine
from agent_forge.exceptions import ExecutionError
from agent_forge.models.base import BaseModel as LLMBaseModel
from agent_forge.monitor import Monitor, MonitorVerdict
from agent_forge.state import StateManager

logger = logging.getLogger(__name__)


class StepStatus(StrEnum):
    """Lifecycle status of a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALIDATED = "invalidated"


class PlanStep(BaseModel):
    """A single executable step within a plan.

    Attributes:
        id: Unique step identifier.
        description: What this step should accomplish.
        action: The action to execute (tool name or LLM prompt).
        action_args: Arguments for the action.
        preconditions: Conditions that must hold before execution.
        postcondition: Condition that should hold after execution.
        expected_state_changes: Predicted state updates after this step.
        depends_on: IDs of steps that must complete first.
        subtasks: Nested sub-steps (for HTN decomposition).
        status: Current execution status.
        result: The step's output after execution.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    action: str = ""
    action_args: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    postcondition: str = ""
    expected_state_changes: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    subtasks: list[PlanStep] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None


class ExecutionResult(BaseModel):
    """Complete result from executing a plan.

    Attributes:
        success: Whether the plan completed successfully.
        degraded: True when no LLM model was available and the planning
            engine fell back to a trivial default plan.
        steps_completed: Number of steps that ran to completion.
        steps_total: Total number of steps in the plan.
        verdicts: Monitor verdicts for each completed step.
        final_state: World state at the end of execution.
        error: Error message if execution failed.
        plan_trace: Complete list of steps in the executed plan.
        backtrack_log: Record of all backtracking events.
        recoverable: Whether the planner should attempt replanning.
    """

    success: bool = False
    degraded: bool = False
    steps_completed: int = 0
    steps_total: int = 0
    verdicts: list[MonitorVerdict] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    plan_trace: list[PlanStep] = Field(default_factory=list)
    backtrack_log: list[dict[str, Any]] = Field(default_factory=list)
    recoverable: bool = True


class Executor:
    """Executes plan steps with monitoring and backtracking support.

    The executor runs each step in dependency order, invokes the
    appropriate tool or LLM call, updates world state, and checks
    postconditions and invariants after each step.

    Args:
        state_manager: Manages world state and checkpoints.
        monitor: Checks postconditions and invariants.
        backtrack_engine: Handles failures by rewinding.
        model: LLM for generating step actions.
        tools: Available tools keyed by name.
        checkpoint_interval: Create a checkpoint every N steps.
    """

    def __init__(
        self,
        state_manager: StateManager,
        monitor: Monitor,
        backtrack_engine: BacktrackEngine,
        model: LLMBaseModel | None = None,
        tools: dict[str, Any] | None = None,
        checkpoint_interval: int = 3,
        system_prompt: str = "",
    ) -> None:
        self._state_manager = state_manager
        self._monitor = monitor
        self._backtrack = backtrack_engine
        self._model = model
        self._tools = tools or {}
        self._checkpoint_interval = checkpoint_interval
        self._system_prompt = system_prompt
        self._steps_since_checkpoint = 0
        logger.debug(
            "Executor initialized (%d tools, checkpoint_interval=%d)",
            len(self._tools),
            checkpoint_interval,
        )

    async def execute_plan(
        self,
        steps: list[PlanStep],
        invariants: list[str],
        max_steps: int | None = None,
    ) -> ExecutionResult:
        """Execute a list of plan steps in order.

        Handles dependency ordering, monitoring, and backtracking.
        Returns a complete ExecutionResult.

        Args:
            steps: Ordered list of plan steps to execute.
            invariants: Global invariant conditions to enforce.

        Returns:
            ExecutionResult with success status and details.
        """
        total_steps = self._count_steps(steps)
        completed = 0
        verdicts: list[MonitorVerdict] = []
        flat_steps = self._flatten_steps(steps)
        pending_steps = [step for step in flat_steps if step.status != StepStatus.INVALIDATED]
        completed_ids: set[str] = set()
        known_step_ids = {step.id for step in pending_steps}

        logger.info("Starting plan execution (%d steps)", total_steps)
        self._state_manager.create_checkpoint(metadata={"reason": "plan_start"})

        while pending_steps:
            if max_steps is not None and completed >= max_steps:
                return ExecutionResult(
                    success=False,
                    steps_completed=completed,
                    steps_total=total_steps,
                    verdicts=verdicts,
                    final_state=self._state_manager.current_state,
                    error=(
                        f"Goal max_steps={max_steps} exceeded after "
                        f"{completed} completed steps"
                    ),
                    recoverable=False,
                )

            next_index: int | None = None
            blocked_step: PlanStep | None = None
            for idx, candidate in enumerate(pending_steps):
                missing = [
                    dep for dep in candidate.depends_on if dep not in completed_ids
                ]
                if not missing:
                    next_index = idx
                    break
                if blocked_step is None:
                    blocked_step = candidate

            if next_index is None:
                missing_dependencies = sorted(
                    {
                        dep
                        for step in pending_steps
                        for dep in step.depends_on
                        if dep not in completed_ids
                    }
                )
                unknown_dependencies = [
                    dep for dep in missing_dependencies if dep not in known_step_ids
                ]
                reason = (
                    f"missing dependencies: {unknown_dependencies}"
                    if unknown_dependencies
                    else "circular or unsatisfied dependencies"
                )
                return ExecutionResult(
                    success=False,
                    steps_completed=completed,
                    steps_total=total_steps,
                    verdicts=verdicts,
                    final_state=self._state_manager.current_state,
                    error=(
                        f"Cannot execute step '{blocked_step.id if blocked_step else ''}': "
                        f"{reason}"
                    ),
                    recoverable=False,
                )

            step = pending_steps.pop(next_index)

            try:
                result = await self._execute_step(step)
                step.result = result
                step.status = StepStatus.COMPLETED

                state_updates = dict(step.expected_state_changes)
                if isinstance(result, dict):
                    state_updates.update(result)
                self._state_manager.update(state_updates)
                self._state_manager.advance_step()
                completed += 1
                completed_ids.add(step.id)

                self._steps_since_checkpoint += 1
                if self._steps_since_checkpoint >= self._checkpoint_interval:
                    self._state_manager.create_checkpoint(metadata={"step_id": step.id})
                    self._steps_since_checkpoint = 0

                for dep_step in flat_steps:
                    if step.id in dep_step.depends_on:
                        self._state_manager.add_dependency(step.id, dep_step.id)

                verdict = await self._monitor.evaluate_step(
                    step_id=step.id,
                    postcondition=step.postcondition,
                    invariants=invariants,
                    step_result=result,
                    expected_state=step.expected_state_changes,
                )
                verdicts.append(verdict)

                if verdict.should_backtrack:
                    if self._backtrack.can_backtrack():
                        event = self._backtrack.backtrack(
                            failed_step_id=step.id,
                            reason=verdict.explanation,
                        )
                        logger.info("Backtracked: %s", event.reason)
                        return ExecutionResult(
                            success=False,
                            steps_completed=completed,
                            steps_total=total_steps,
                            verdicts=verdicts,
                            final_state=self._state_manager.current_state,
                            error=f"Backtracked at step '{step.id}': {verdict.explanation}",
                            recoverable=True,
                        )
                    else:
                        return ExecutionResult(
                            success=False,
                            steps_completed=completed,
                            steps_total=total_steps,
                            verdicts=verdicts,
                            final_state=self._state_manager.current_state,
                            error=f"Cannot backtrack further at step '{step.id}'",
                            recoverable=False,
                        )
                else:
                    self._backtrack.reset_depth()

            except ExecutionError as exc:
                step.status = StepStatus.FAILED
                logger.error("Step '%s' failed: %s", step.id, exc)
                return ExecutionResult(
                    success=False,
                    steps_completed=completed,
                    steps_total=total_steps,
                    verdicts=verdicts,
                    final_state=self._state_manager.current_state,
                    error=str(exc),
                    recoverable=exc.recoverable,
                )

        logger.info(
            "Plan execution complete: %d/%d steps succeeded",
            completed,
            total_steps,
        )
        return ExecutionResult(
            success=True,
            steps_completed=completed,
            steps_total=total_steps,
            verdicts=verdicts,
            final_state=self._state_manager.current_state,
        )

    async def _execute_step(self, step: PlanStep) -> Any:
        """Execute a single plan step.

        Routes to tool execution or LLM generation based on whether
        the step's action matches a known tool name.

        Args:
            step: The plan step to execute.

        Returns:
            The step's result.

        Raises:
            ExecutionError: If the step fails.
        """
        step.status = StepStatus.RUNNING
        logger.info("Executing step '%s': %s", step.id, step.description[:80])

        if step.action in self._tools:
            tool = self._tools[step.action]
            try:
                return await tool.execute(**step.action_args)
            except Exception as exc:
                raise ExecutionError(
                    message=str(exc),
                    step_id=step.id,
                    recoverable=True,
                ) from exc

        if self._model is not None:
            prompt = self._build_step_prompt(step)
            response = await self._model.generate_text(
                prompt,
                system=self._system_prompt or None,
                tools=self._tool_schemas(),
            )
            return response

        return f"Completed: {step.description}"

    def _build_step_prompt(self, step: PlanStep) -> str:
        """Build an LLM prompt for executing a step.

        Args:
            step: The step to generate a prompt for.

        Returns:
            Formatted prompt string.
        """
        state_summary = self._state_manager.current_state
        return (
            f"Execute the following step:\n\n"
            f"Description: {step.description}\n"
            f"Action: {step.action}\n"
            f"Arguments: {step.action_args}\n"
            f"Current state: {state_summary}\n\n"
            f"Provide the result of executing this step."
        )

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible schemas for all registered tools."""
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            to_schema = getattr(tool, "to_schema", None)
            if callable(to_schema):
                schemas.append(to_schema())
        return schemas

    @staticmethod
    def _flatten_steps(steps: list[PlanStep]) -> list[PlanStep]:
        """Flatten a tree of steps into dependency-ordered list.

        Args:
            steps: Top-level steps (may contain subtasks).

        Returns:
            Flat, ordered list of all steps.
        """
        flat: list[PlanStep] = []
        for step in steps:
            if step.subtasks:
                flat.extend(Executor._flatten_steps(step.subtasks))
            flat.append(step)
        return flat

    @staticmethod
    def _count_steps(steps: list[PlanStep]) -> int:
        """Count total steps including subtasks.

        Args:
            steps: Top-level steps.

        Returns:
            Total number of steps.
        """
        count = 0
        for step in steps:
            count += 1
            if step.subtasks:
                count += Executor._count_steps(step.subtasks)
        return count
