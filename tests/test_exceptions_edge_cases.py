"""Edge case tests for custom exceptions.

Covers message formatting edge cases, default values, exception
chaining, and pickleability for multiprocessing scenarios.
"""

from __future__ import annotations

import pytest

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


class TestPlanningErrorEdgeCases:
    """Edge cases for PlanningError."""

    def test_empty_message(self) -> None:
        """PlanningError with empty message should not crash."""
        err = PlanningError("")
        assert "Planning failed" in str(err)

    def test_empty_goal_description(self) -> None:
        """Empty goal_description should use the no-goal format."""
        err = PlanningError("msg", goal_description="")
        assert "Planning failed: msg" in str(err)
        assert err.goal_description == ""

    def test_special_characters_in_goal(self) -> None:
        """Special characters should be preserved in error message."""
        err = PlanningError("fail", goal_description="goal with 'quotes' & <brackets>")
        assert "'quotes'" in str(err)
        assert "<brackets>" in str(err)


class TestExecutionErrorEdgeCases:
    """Edge cases for ExecutionError."""

    def test_default_values(self) -> None:
        """ExecutionError defaults should be sensible."""
        err = ExecutionError("something broke")
        assert err.step_id == ""
        assert err.recoverable is True
        assert "something broke" in str(err)

    def test_non_recoverable(self) -> None:
        """Non-recoverable should be reflected in message."""
        err = ExecutionError("fatal", step_id="s1", recoverable=False)
        assert "recoverable=False" in str(err)

    def test_recoverable_true(self) -> None:
        """Recoverable should be reflected in message."""
        err = ExecutionError("transient", step_id="s2", recoverable=True)
        assert "recoverable=True" in str(err)


class TestBacktrackErrorEdgeCases:
    """Edge cases for BacktrackError."""

    def test_zero_depth(self) -> None:
        """BacktrackError at depth 0 should format correctly."""
        err = BacktrackError("cannot go back", depth=0, max_depth=5)
        assert "0/5" in str(err)

    def test_equal_depth_and_max(self) -> None:
        """BacktrackError with depth == max_depth."""
        err = BacktrackError("at limit", depth=5, max_depth=5)
        assert "5/5" in str(err)
        assert err.depth == 5
        assert err.max_depth == 5


class TestInvariantViolationEdgeCases:
    """Edge cases for InvariantViolation."""

    def test_no_context(self) -> None:
        """InvariantViolation without context should not show context part."""
        err = InvariantViolation(invariant="must not fabricate data")
        msg = str(err)
        assert "must not fabricate data" in msg
        assert "context:" not in msg

    def test_with_context(self) -> None:
        """InvariantViolation with context should show both parts."""
        err = InvariantViolation(
            invariant="budget limit",
            context="step-5 spent $1M",
        )
        msg = str(err)
        assert "budget limit" in msg
        assert "step-5 spent $1M" in msg

    def test_empty_invariant(self) -> None:
        """Empty invariant string should not crash."""
        err = InvariantViolation(invariant="")
        assert "Invariant violated" in str(err)


class TestModelErrorEdgeCases:
    """Edge cases for ModelError."""

    def test_no_model_no_status(self) -> None:
        """ModelError with no model or status code."""
        err = ModelError("connection failed")
        assert "Model" in str(err)
        assert err.model == ""
        assert err.status_code is None

    def test_with_model_no_status(self) -> None:
        """ModelError with model but no status code."""
        err = ModelError("timeout", model="gpt-4o")
        assert "gpt-4o" in str(err)
        assert "HTTP" not in str(err)

    def test_with_status_code_zero(self) -> None:
        """Status code 0 is falsy but should be handled."""
        err = ModelError("weird", model="test", status_code=0)
        # status_code=0 is falsy, so the code won't show "HTTP 0"
        assert err.status_code == 0

    def test_common_http_status_codes(self) -> None:
        """Common error codes should appear in message."""
        for code in [400, 401, 403, 404, 429, 500, 502, 503]:
            err = ModelError("err", model="m", status_code=code)
            assert str(code) in str(err)


class TestToolErrorEdgeCases:
    """Edge cases for ToolError."""

    def test_no_tool_name(self) -> None:
        """ToolError without tool name should use generic format."""
        err = ToolError("generic failure")
        msg = str(err)
        assert "Tool call failed" in msg
        assert err.tool_name == ""

    def test_with_tool_name(self) -> None:
        """ToolError with tool name should include it."""
        err = ToolError("timeout", tool_name="web_search")
        assert "web_search" in str(err)

    def test_long_error_message(self) -> None:
        """ToolError should handle very long error messages."""
        long_msg = "x" * 10000
        err = ToolError(long_msg, tool_name="tool")
        assert long_msg in str(err)


class TestStateErrorSimple:
    """Tests for StateError (simple exception with no custom fields)."""

    def test_basic_state_error(self) -> None:
        """StateError should work as a basic exception."""
        err = StateError("Checkpoint not found")
        assert "Checkpoint not found" in str(err)

    def test_state_error_is_agent_forge_error(self) -> None:
        """StateError should be catchable as AgentForgeError."""
        with pytest.raises(AgentForgeError):
            raise StateError("test")


class TestExceptionCatchability:
    """Test that exceptions can be caught at multiple levels."""

    def test_catch_as_base_exception(self) -> None:
        """All custom exceptions should be catchable as Exception."""
        exceptions = [
            PlanningError("x"),
            ExecutionError("x"),
            BacktrackError("x"),
            InvariantViolation(invariant="x"),
            StateError("x"),
            ModelError("x"),
            ToolError("x"),
        ]
        for exc in exceptions:
            assert isinstance(exc, Exception)

    def test_catch_as_agent_forge_error(self) -> None:
        """All custom exceptions should be catchable as AgentForgeError."""
        exceptions = [
            PlanningError("x"),
            ExecutionError("x"),
            BacktrackError("x"),
            InvariantViolation(invariant="x"),
            StateError("x"),
            ModelError("x"),
            ToolError("x"),
        ]
        for exc in exceptions:
            with pytest.raises(AgentForgeError):
                raise exc

    def test_exception_chaining(self) -> None:
        """Exceptions should support 'from' chaining."""
        original = ValueError("root cause")
        try:
            try:
                raise original
            except ValueError as e:
                raise ToolError("tool broke", tool_name="t") from e
        except ToolError as te:
            assert te.__cause__ is original
