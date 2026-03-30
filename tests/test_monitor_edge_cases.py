"""Edge case tests for the Monitor module.

Covers postcondition evaluation with various result types,
invariant checking with LLM, drift detection edge cases,
and evaluate_step integration scenarios.
"""

from __future__ import annotations

import pytest

from agent_forge.monitor import CheckResult, Monitor
from agent_forge.state import StateManager
from tests.conftest import MockModel


class TestHeuristicPostconditionEdgeCases:
    """Edge cases for heuristic postcondition checking."""

    def test_heuristic_true_bool(self) -> None:
        """True boolean should pass."""
        result = Monitor._heuristic_postcondition_check("condition", True)
        assert result == CheckResult.PASSED

    def test_heuristic_zero_int(self) -> None:
        """Zero int is falsy but is not None/bool/empty-string, so should pass."""
        result = Monitor._heuristic_postcondition_check("condition", 0)
        assert result == CheckResult.PASSED

    def test_heuristic_empty_list(self) -> None:
        """Empty list is falsy but is not None/bool/str, so should pass."""
        result = Monitor._heuristic_postcondition_check("condition", [])
        assert result == CheckResult.PASSED

    def test_heuristic_nonempty_string(self) -> None:
        """Non-empty string should pass."""
        result = Monitor._heuristic_postcondition_check("condition", "result data")
        assert result == CheckResult.PASSED

    def test_heuristic_whitespace_only_string(self) -> None:
        """Whitespace-only strings should fail (treated as empty)."""
        result = Monitor._heuristic_postcondition_check("cond", "   \t\n  ")
        assert result == CheckResult.FAILED

    def test_heuristic_dict_result(self) -> None:
        """Non-empty dict should pass."""
        result = Monitor._heuristic_postcondition_check("cond", {"key": "val"})
        assert result == CheckResult.PASSED

    def test_heuristic_empty_dict(self) -> None:
        """Empty dict is falsy but not None/bool/str, so should pass."""
        result = Monitor._heuristic_postcondition_check("cond", {})
        assert result == CheckResult.PASSED


class TestPostconditionWithModel:
    """Tests for postcondition checking with LLM model."""

    @pytest.mark.asyncio
    async def test_model_returns_passed(self) -> None:
        """Model returning 'PASSED' should yield PASSED."""
        sm = StateManager()
        model = MockModel(responses=["PASSED"])
        monitor = Monitor(state_manager=sm, model=model)

        result = await monitor.check_postcondition("s1", "condition holds", "result")
        assert result == CheckResult.PASSED

    @pytest.mark.asyncio
    async def test_model_returns_failed(self) -> None:
        """Model returning 'FAILED' should yield FAILED."""
        sm = StateManager()
        model = MockModel(responses=["FAILED"])
        monitor = Monitor(state_manager=sm, model=model)

        result = await monitor.check_postcondition("s1", "condition broken", None)
        assert result == CheckResult.FAILED

    @pytest.mark.asyncio
    async def test_model_returns_uncertain(self) -> None:
        """Model returning 'UNCERTAIN' should yield UNCERTAIN."""
        sm = StateManager()
        model = MockModel(responses=["UNCERTAIN"])
        monitor = Monitor(state_manager=sm, model=model)

        result = await monitor.check_postcondition("s1", "maybe?", "partial")
        assert result == CheckResult.UNCERTAIN

    @pytest.mark.asyncio
    async def test_model_returns_lowercase_passed(self) -> None:
        """Model returning lowercase 'passed' should still match."""
        sm = StateManager()
        model = MockModel(responses=["passed"])
        monitor = Monitor(state_manager=sm, model=model)

        result = await monitor.check_postcondition("s1", "condition", "ok")
        assert result == CheckResult.PASSED

    @pytest.mark.asyncio
    async def test_model_returns_garbage(self) -> None:
        """Model returning unrecognized text should default to UNCERTAIN."""
        sm = StateManager()
        model = MockModel(responses=["I don't know what you mean"])
        monitor = Monitor(state_manager=sm, model=model)

        result = await monitor.check_postcondition("s1", "condition", "ok")
        assert result == CheckResult.UNCERTAIN

    @pytest.mark.asyncio
    async def test_model_returns_passed_with_extra_text(self) -> None:
        """Model with 'PASSED' embedded in text should still match."""
        sm = StateManager()
        model = MockModel(responses=["The condition has PASSED successfully"])
        monitor = Monitor(state_manager=sm, model=model)

        result = await monitor.check_postcondition("s1", "condition", "ok")
        assert result == CheckResult.PASSED


class TestInvariantCheckingEdgeCases:
    """Edge cases for invariant checking."""

    @pytest.mark.asyncio
    async def test_empty_invariants_list(self) -> None:
        """Empty invariants list should return empty dict."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        results = await monitor.check_invariants("step-1", [])
        assert results == {}

    @pytest.mark.asyncio
    async def test_many_invariants_no_model(self) -> None:
        """Without model, all invariants should pass."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        invariants = [f"invariant-{i}" for i in range(20)]
        results = await monitor.check_invariants("step-1", invariants)
        assert len(results) == 20
        assert all(r == CheckResult.PASSED for r in results.values())

    @pytest.mark.asyncio
    async def test_invariant_uncertain_with_model(self) -> None:
        """Model returning UNCERTAIN should yield CheckResult.UNCERTAIN."""
        sm = StateManager()
        model = MockModel(responses=["UNCERTAIN"])
        monitor = Monitor(state_manager=sm, model=model)

        results = await monitor.check_invariants("step-1", ["ambiguous rule"])
        assert results["ambiguous rule"] == CheckResult.UNCERTAIN


class TestDriftDetectionEdgeCases:
    """Edge cases for state drift computation."""

    def test_drift_all_keys_match(self) -> None:
        """Drift should be 0 when all expected keys match."""
        sm = StateManager()
        sm.set("a", 1)
        sm.set("b", "hello")
        sm.set("c", [1, 2, 3])
        monitor = Monitor(state_manager=sm)

        drift = monitor.compute_drift({"a": 1, "b": "hello", "c": [1, 2, 3]})
        assert drift == 0.0

    def test_drift_single_key_mismatch(self) -> None:
        """Drift with one mismatch out of many keys."""
        sm = StateManager()
        sm.set("a", 1)
        sm.set("b", 2)
        sm.set("c", 3)
        sm.set("d", 4)
        monitor = Monitor(state_manager=sm)

        drift = monitor.compute_drift({"a": 1, "b": 2, "c": 3, "d": 999})
        assert drift == pytest.approx(0.25)

    def test_drift_expected_key_missing_from_state(self) -> None:
        """Keys expected but missing from state should count as drift."""
        sm = StateManager()
        sm.set("a", 1)
        monitor = Monitor(state_manager=sm)

        drift = monitor.compute_drift({"a": 1, "missing": "value"})
        assert drift == pytest.approx(0.5)

    def test_drift_with_none_values(self) -> None:
        """None in state matching None in expected should be 0 drift."""
        sm = StateManager()
        sm.set("key", None)
        monitor = Monitor(state_manager=sm)

        drift = monitor.compute_drift({"key": None})
        assert drift == 0.0

    def test_drift_type_mismatch(self) -> None:
        """Different types for same key should count as drift."""
        sm = StateManager()
        sm.set("val", "123")
        monitor = Monitor(state_manager=sm)

        drift = monitor.compute_drift({"val": 123})
        assert drift == 1.0


class TestEvaluateStepEdgeCases:
    """Edge cases for the full evaluate_step pipeline."""

    @pytest.mark.asyncio
    async def test_evaluate_step_no_postcondition_no_invariants(self) -> None:
        """Step with no postcondition and no invariants should pass."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)

        verdict = await monitor.evaluate_step(
            step_id="s1",
            postcondition="",
            invariants=[],
            step_result="anything",
        )
        assert not verdict.should_backtrack
        assert verdict.postcondition_result == CheckResult.PASSED
        assert verdict.drift_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_step_drift_exactly_at_threshold(self) -> None:
        """Drift exactly at threshold should NOT trigger backtrack (> not >=)."""
        sm = StateManager()
        sm.set("a", 1)
        sm.set("b", "wrong")
        monitor = Monitor(state_manager=sm, drift_threshold=0.5)

        verdict = await monitor.evaluate_step(
            step_id="s1",
            postcondition="",
            invariants=[],
            step_result="ok",
            expected_state={"a": 1, "b": "right"},
        )
        # drift is 0.5 == threshold, should_backtrack uses >
        assert not verdict.should_backtrack

    @pytest.mark.asyncio
    async def test_evaluate_step_drift_just_above_threshold(self) -> None:
        """Drift just above threshold should trigger backtrack."""
        sm = StateManager()
        sm.set("a", "wrong")
        sm.set("b", "wrong")
        sm.set("c", "correct")
        monitor = Monitor(state_manager=sm, drift_threshold=0.6)

        verdict = await monitor.evaluate_step(
            step_id="s1",
            postcondition="",
            invariants=[],
            step_result="ok",
            expected_state={"a": "right", "b": "right", "c": "correct"},
        )
        # 2/3 keys wrong ~ 0.6667 > 0.6 threshold
        assert verdict.should_backtrack
        assert "drift" in verdict.explanation

    @pytest.mark.asyncio
    async def test_evaluate_step_failed_invariant_with_model(self) -> None:
        """Failed invariant via model should trigger backtrack."""
        sm = StateManager()
        model = MockModel(responses=["PASSED", "FAILED"])  # postcondition, invariant
        monitor = Monitor(state_manager=sm, model=model)

        verdict = await monitor.evaluate_step(
            step_id="s1",
            postcondition="some condition",
            invariants=["safety rule"],
            step_result="ok",
        )
        assert verdict.should_backtrack
        assert "invariants violated" in verdict.explanation

    @pytest.mark.asyncio
    async def test_evaluate_step_multiple_failures(self) -> None:
        """Multiple failure types should all be recorded in explanation."""
        sm = StateManager()
        model = MockModel(responses=["FAILED", "FAILED"])  # postcondition, invariant
        monitor = Monitor(state_manager=sm, model=model, drift_threshold=0.0)

        # Set up for drift (expected_state won't match)
        verdict = await monitor.evaluate_step(
            step_id="s1",
            postcondition="must pass",
            invariants=["safety"],
            step_result="ok",
            expected_state={"key": "expected"},
        )
        assert verdict.should_backtrack
        assert "postcondition failed" in verdict.explanation

    @pytest.mark.asyncio
    async def test_violations_accumulate(self) -> None:
        """Multiple failing evaluations should all be recorded."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)

        for i in range(5):
            await monitor.evaluate_step(
                step_id=f"step-{i}",
                postcondition="must have result",
                invariants=[],
                step_result=None,  # Will fail heuristic
            )

        assert len(monitor.violations) == 5
        violation_ids = [v.step_id for v in monitor.violations]
        for i in range(5):
            assert f"step-{i}" in violation_ids


class TestMonitorStateSummary:
    """Tests for the state summarization helper."""

    def test_summarize_empty_state(self) -> None:
        """Empty state should produce '(empty state)' summary."""
        sm = StateManager()
        monitor = Monitor(state_manager=sm)
        summary = monitor._summarize_state()
        assert summary == "(empty state)"

    def test_summarize_nonempty_state(self) -> None:
        """Non-empty state should include key-value pairs."""
        sm = StateManager(initial_state={"name": "test", "count": 42})
        monitor = Monitor(state_manager=sm)
        summary = monitor._summarize_state()
        assert "name" in summary
        assert "count" in summary

    def test_summarize_truncates_long_values(self) -> None:
        """Long values should be truncated in summary."""
        sm = StateManager()
        sm.set("long_key", "x" * 200)
        monitor = Monitor(state_manager=sm)
        summary = monitor._summarize_state()
        # repr(value)[:100] should truncate
        assert len(summary) < 200
