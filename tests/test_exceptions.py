"""Tests for custom exceptions."""

from __future__ import annotations

from agent_forge.exceptions import (
    AgentForgeError,
    BacktrackError,
    ExecutionError,
    InvariantViolation,
    ModelError,
    PlanningError,
    StateError,
    ToolError,
)


class TestExceptions:
    """Tests for custom exception classes."""

    def test_planning_error(self) -> None:
        """PlanningError should format message with goal."""
        err = PlanningError("no valid plan", goal_description="write report")
        assert "write report" in str(err)
        assert "no valid plan" in str(err)

    def test_planning_error_no_goal(self) -> None:
        """PlanningError should work without goal description."""
        err = PlanningError("generic failure")
        assert "generic failure" in str(err)

    def test_execution_error(self) -> None:
        """ExecutionError should include step ID and recoverability."""
        err = ExecutionError("timeout", step_id="step-5", recoverable=False)
        assert err.step_id == "step-5"
        assert err.recoverable is False
        assert "step-5" in str(err)

    def test_backtrack_error(self) -> None:
        """BacktrackError should include depth info."""
        err = BacktrackError("too deep", depth=3, max_depth=5)
        assert err.depth == 3
        assert err.max_depth == 5
        assert "3/5" in str(err)

    def test_invariant_violation(self) -> None:
        """InvariantViolation should include the violated invariant."""
        err = InvariantViolation(
            invariant="no data fabrication",
            context="step-3 output",
        )
        assert "no data fabrication" in str(err)
        assert "step-3 output" in str(err)

    def test_model_error(self) -> None:
        """ModelError should include model name and status code."""
        err = ModelError("rate limited", model="gpt-4o", status_code=429)
        assert err.model == "gpt-4o"
        assert err.status_code == 429
        assert "429" in str(err)

    def test_tool_error(self) -> None:
        """ToolError should include tool name."""
        err = ToolError("not found", tool_name="search")
        assert "search" in str(err)

    def test_inheritance(self) -> None:
        """All exceptions should inherit from AgentForgeError."""
        assert issubclass(PlanningError, AgentForgeError)
        assert issubclass(ExecutionError, AgentForgeError)
        assert issubclass(BacktrackError, AgentForgeError)
        assert issubclass(InvariantViolation, AgentForgeError)
        assert issubclass(StateError, AgentForgeError)
        assert issubclass(ModelError, AgentForgeError)
        assert issubclass(ToolError, AgentForgeError)
        assert issubclass(AgentForgeError, Exception)
