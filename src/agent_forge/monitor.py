"""Execution monitor with invariant checking and drift detection.

The Monitor watches each execution step and evaluates:
1. Postconditions: did the step achieve what it claimed?
2. Invariants: are global safety constraints still satisfied?
3. Drift: has the world state diverged from the plan's expectations?

When any check fails, the monitor signals the backtracking engine
to rewind and replan.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_forge.models.base import BaseModel as LLMBaseModel
    from agent_forge.state import StateManager

logger = logging.getLogger(__name__)


class CheckResult(StrEnum):
    """Outcome of a monitoring check."""

    PASSED = "passed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class MonitorVerdict(BaseModel):
    """Result of monitoring a single step.

    Attributes:
        step_id: The step that was monitored.
        postcondition_result: Whether the step's postcondition held.
        invariant_results: Per-invariant check results.
        drift_score: Numeric measure of state drift (0.0 = no drift).
        should_backtrack: Whether backtracking is recommended.
        explanation: Human-readable summary of the verdict.
    """

    step_id: str
    postcondition_result: CheckResult = CheckResult.PASSED
    invariant_results: dict[str, CheckResult] = Field(default_factory=dict)
    drift_score: float = 0.0
    should_backtrack: bool = False
    explanation: str = ""


class Monitor:
    """Monitors execution steps for correctness and safety.

    After each step, the monitor uses the LLM to evaluate postconditions
    and invariants against the current world state. It also computes a
    drift score measuring how far the actual state has diverged from
    the planned state.

    Args:
        state_manager: The state manager tracking world state.
        model: LLM used for evaluating natural-language conditions.
        drift_threshold: Maximum drift score before triggering backtrack.
    """

    def __init__(
        self,
        state_manager: StateManager,
        model: LLMBaseModel | None = None,
        drift_threshold: float = 0.7,
        system_prompt: str = "",
    ) -> None:
        self._state_manager = state_manager
        self._model = model
        self._drift_threshold = drift_threshold
        self._system_prompt = system_prompt
        self._violations: list[MonitorVerdict] = []
        logger.debug("Monitor initialized (drift_threshold=%.2f)", drift_threshold)

    @property
    def violations(self) -> list[MonitorVerdict]:
        """Return all recorded invariant violations."""
        return list(self._violations)

    async def check_postcondition(
        self,
        step_id: str,
        postcondition: str,
        step_result: Any,
    ) -> CheckResult:
        """Evaluate whether a step's postcondition holds.

        Uses the LLM to assess the postcondition against the step's
        actual result and current world state.

        Args:
            step_id: Identifier of the completed step.
            postcondition: Natural-language postcondition to check.
            step_result: The actual output/result of the step.

        Returns:
            CheckResult indicating pass, fail, or uncertain.
        """
        if not postcondition:
            return CheckResult.PASSED

        if self._model is None:
            return self._heuristic_postcondition_check(postcondition, step_result)

        state_summary = self._summarize_state()
        prompt = (
            f"You are evaluating whether a postcondition holds after an action.\n\n"
            f"Postcondition: {postcondition}\n"
            f"Step result: {step_result}\n"
            f"Current state: {state_summary}\n\n"
            f"Does the postcondition hold? Answer EXACTLY one word: PASSED, FAILED, or UNCERTAIN."
        )

        response = await self._model.generate_text(
            prompt,
            system=self._system_prompt or None,
        )
        cleaned = response.strip().upper()
        if "PASSED" in cleaned:
            return CheckResult.PASSED
        elif "FAILED" in cleaned:
            return CheckResult.FAILED
        return CheckResult.UNCERTAIN

    async def check_invariants(
        self,
        step_id: str,
        invariants: list[str],
    ) -> dict[str, CheckResult]:
        """Check all global invariants against the current state.

        Args:
            step_id: Identifier of the step just completed.
            invariants: List of natural-language invariant conditions.

        Returns:
            Dict mapping each invariant to its CheckResult.
        """
        results: dict[str, CheckResult] = {}

        for invariant in invariants:
            if self._model is None:
                results[invariant] = CheckResult.PASSED
                continue

            state_summary = self._summarize_state()
            prompt = (
                f"You are a safety monitor. Check if this invariant still holds.\n\n"
                f"Invariant: {invariant}\n"
                f"Current world state: {state_summary}\n\n"
                f"Does this invariant hold? Answer EXACTLY one word: PASSED, FAILED, or UNCERTAIN."
            )
            response = await self._model.generate_text(
                prompt,
                system=self._system_prompt or None,
            )
            cleaned = response.strip().upper()

            if "FAILED" in cleaned:
                results[invariant] = CheckResult.FAILED
                logger.warning("Invariant FAILED at step %s: %s", step_id, invariant)
            elif "UNCERTAIN" in cleaned:
                results[invariant] = CheckResult.UNCERTAIN
            else:
                results[invariant] = CheckResult.PASSED

        return results

    def compute_drift(
        self,
        expected_state: dict[str, Any],
    ) -> float:
        """Compute how far the actual state has drifted from expectations.

        Measures the fraction of expected state keys whose values
        differ from the actual state.

        Args:
            expected_state: The state the plan predicted after this step.

        Returns:
            Drift score between 0.0 (no drift) and 1.0 (total drift).
        """
        if not expected_state:
            return 0.0

        current = self._state_manager.current_state
        mismatches = 0
        total = len(expected_state)

        for key, expected_value in expected_state.items():
            actual_value = current.get(key)
            if actual_value != expected_value:
                mismatches += 1
                logger.debug(
                    "Drift detected: key='%s' expected=%s actual=%s",
                    key,
                    repr(expected_value)[:50],
                    repr(actual_value)[:50],
                )

        score = mismatches / total if total > 0 else 0.0
        return round(score, 4)

    async def evaluate_step(
        self,
        step_id: str,
        postcondition: str,
        invariants: list[str],
        step_result: Any,
        expected_state: dict[str, Any] | None = None,
    ) -> MonitorVerdict:
        """Run all monitoring checks on a completed step.

        This is the main entry point called by the executor after
        each step completes.

        Args:
            step_id: Identifier of the completed step.
            postcondition: The step's postcondition to verify.
            invariants: Global invariants to check.
            step_result: The step's output/result.
            expected_state: The state the plan predicted. Used for drift.

        Returns:
            MonitorVerdict with all check results and recommendation.
        """
        post_result = await self.check_postcondition(
            step_id, postcondition, step_result
        )
        inv_results = await self.check_invariants(step_id, invariants)
        drift_score = self.compute_drift(expected_state or {})

        should_backtrack = (
            post_result == CheckResult.FAILED
            or any(r == CheckResult.FAILED for r in inv_results.values())
            or drift_score > self._drift_threshold
        )

        reasons: list[str] = []
        if post_result == CheckResult.FAILED:
            reasons.append("postcondition failed")
        failed_invariants = [
            inv for inv, r in inv_results.items() if r == CheckResult.FAILED
        ]
        if failed_invariants:
            reasons.append(f"invariants violated: {failed_invariants}")
        if drift_score > self._drift_threshold:
            reasons.append(
                f"drift={drift_score:.2f} > threshold={self._drift_threshold:.2f}"
            )

        verdict = MonitorVerdict(
            step_id=step_id,
            postcondition_result=post_result,
            invariant_results=inv_results,
            drift_score=drift_score,
            should_backtrack=should_backtrack,
            explanation="; ".join(reasons) if reasons else "all checks passed",
        )

        if should_backtrack:
            self._violations.append(verdict)
            logger.warning(
                "Step %s triggered backtrack: %s", step_id, verdict.explanation
            )
        else:
            logger.info("Step %s passed monitoring: %s", step_id, verdict.explanation)

        return verdict

    def _summarize_state(self) -> str:
        """Create a concise text summary of the current world state."""
        state = self._state_manager.current_state
        if not state:
            return "(empty state)"
        parts = [f"{k}: {repr(v)[:100]}" for k, v in list(state.items())[:20]]
        return "\n".join(parts)

    @staticmethod
    def _heuristic_postcondition_check(
        postcondition: str, step_result: Any
    ) -> CheckResult:
        """Simple heuristic check when no LLM is available.

        Args:
            postcondition: The condition to evaluate.
            step_result: The step's result.

        Returns:
            PASSED if step_result is truthy, FAILED otherwise.
        """
        if step_result is None:
            return CheckResult.FAILED
        if isinstance(step_result, bool):
            return CheckResult.PASSED if step_result else CheckResult.FAILED
        if isinstance(step_result, str) and not step_result.strip():
            return CheckResult.FAILED
        return CheckResult.PASSED
