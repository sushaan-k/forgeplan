"""Tests for the Monitor module."""

from __future__ import annotations

import pytest

from agent_forge.monitor import CheckResult, Monitor, MonitorVerdict
from agent_forge.state import StateManager
from tests.conftest import MockModel


class TestMonitor:
    """Tests for execution monitoring."""

    def test_heuristic_postcondition_truthy(self, monitor: Monitor) -> None:
        """Truthy results should pass heuristic check."""
        result = Monitor._heuristic_postcondition_check("condition", "result")
        assert result == CheckResult.PASSED

    def test_heuristic_postcondition_none(self, monitor: Monitor) -> None:
        """None results should fail heuristic check."""
        result = Monitor._heuristic_postcondition_check("condition", None)
        assert result == CheckResult.FAILED

    def test_heuristic_postcondition_false(self, monitor: Monitor) -> None:
        """False results should fail heuristic check."""
        result = Monitor._heuristic_postcondition_check("condition", False)
        assert result == CheckResult.FAILED

    def test_heuristic_postcondition_empty_string(self, monitor: Monitor) -> None:
        """Empty string results should fail heuristic check."""
        result = Monitor._heuristic_postcondition_check("condition", "  ")
        assert result == CheckResult.FAILED

    @pytest.mark.asyncio
    async def test_check_postcondition_no_model(self, monitor: Monitor) -> None:
        """Should use heuristic when no LLM is available."""
        result = await monitor.check_postcondition("step-1", "must succeed", "ok")
        assert result == CheckResult.PASSED

    @pytest.mark.asyncio
    async def test_check_postcondition_empty(self, monitor: Monitor) -> None:
        """Empty postcondition should always pass."""
        result = await monitor.check_postcondition("step-1", "", None)
        assert result == CheckResult.PASSED

    @pytest.mark.asyncio
    async def test_check_invariants_no_model(self, monitor: Monitor) -> None:
        """Invariants should all pass when no LLM is available."""
        results = await monitor.check_invariants("step-1", ["inv1", "inv2"])
        assert all(r == CheckResult.PASSED for r in results.values())

    @pytest.mark.asyncio
    async def test_check_invariants_with_model(
        self, state_manager: StateManager
    ) -> None:
        """Invariants should be evaluated by the LLM."""
        model = MockModel(responses=["PASSED", "FAILED"])
        monitor = Monitor(state_manager=state_manager, model=model)

        results = await monitor.check_invariants(
            "step-1", ["good invariant", "bad invariant"]
        )
        assert results["good invariant"] == CheckResult.PASSED
        assert results["bad invariant"] == CheckResult.FAILED


class TestDriftDetection:
    """Tests for state drift computation."""

    def test_no_drift(self, state_manager: StateManager) -> None:
        """Zero drift when state matches expectations."""
        state_manager.set("x", 1)
        state_manager.set("y", 2)
        monitor = Monitor(state_manager=state_manager)

        drift = monitor.compute_drift({"x": 1, "y": 2})
        assert drift == 0.0

    def test_full_drift(self, state_manager: StateManager) -> None:
        """Full drift when nothing matches."""
        state_manager.set("x", 99)
        monitor = Monitor(state_manager=state_manager)

        drift = monitor.compute_drift({"x": 1, "y": 2})
        assert drift == 1.0

    def test_partial_drift(self, state_manager: StateManager) -> None:
        """Partial drift when some keys differ."""
        state_manager.set("x", 1)
        state_manager.set("y", 99)
        monitor = Monitor(state_manager=state_manager)

        drift = monitor.compute_drift({"x": 1, "y": 2})
        assert drift == 0.5

    def test_empty_expected_state(self, monitor: Monitor) -> None:
        """Empty expected state should yield zero drift."""
        drift = monitor.compute_drift({})
        assert drift == 0.0


class TestEvaluateStep:
    """Tests for the full step evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_evaluate_passing_step(self, monitor: Monitor) -> None:
        """Step should pass when postcondition holds and no drift."""
        verdict = await monitor.evaluate_step(
            step_id="step-1",
            postcondition="should succeed",
            invariants=[],
            step_result="done",
            expected_state={},
        )
        assert isinstance(verdict, MonitorVerdict)
        assert not verdict.should_backtrack
        assert verdict.postcondition_result == CheckResult.PASSED

    @pytest.mark.asyncio
    async def test_evaluate_failing_postcondition(self, monitor: Monitor) -> None:
        """Step should trigger backtrack when postcondition fails."""
        verdict = await monitor.evaluate_step(
            step_id="step-1",
            postcondition="should succeed",
            invariants=[],
            step_result=None,
        )
        assert verdict.should_backtrack
        assert verdict.postcondition_result == CheckResult.FAILED

    @pytest.mark.asyncio
    async def test_evaluate_high_drift(self, state_manager: StateManager) -> None:
        """Step should trigger backtrack when drift exceeds threshold."""
        state_manager.set("key", "wrong")
        monitor = Monitor(state_manager=state_manager, drift_threshold=0.3)

        verdict = await monitor.evaluate_step(
            step_id="step-1",
            postcondition="",
            invariants=[],
            step_result="ok",
            expected_state={"key": "right", "other": "value"},
        )
        assert verdict.should_backtrack
        assert verdict.drift_score > 0.3

    @pytest.mark.asyncio
    async def test_violations_recorded(self, monitor: Monitor) -> None:
        """Violations should be recorded for later inspection."""
        assert monitor.violations == []

        await monitor.evaluate_step(
            step_id="step-1",
            postcondition="must succeed",
            invariants=[],
            step_result=None,
        )
        assert len(monitor.violations) == 1
        assert monitor.violations[0].step_id == "step-1"
