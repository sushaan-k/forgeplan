"""Backtracking engine with causal dependency reasoning.

When the monitor detects a failure, the backtracking engine:
1. Identifies which checkpoint to rewind to.
2. Invalidates all causally-dependent downstream steps.
3. Signals the planner to replan from the restored state.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from agent_forge.exceptions import BacktrackError
from agent_forge.state import StateManager

logger = logging.getLogger(__name__)


class BacktrackEvent(BaseModel):
    """Record of a single backtracking event.

    Attributes:
        trigger_step_id: The step whose failure triggered backtracking.
        checkpoint_id: The checkpoint that was restored.
        invalidated_steps: Steps that were invalidated by the backtrack.
        reason: Why backtracking was triggered.
        depth: How many times we've backtracked in sequence.
    """

    trigger_step_id: str
    checkpoint_id: str
    invalidated_steps: list[str]
    reason: str
    depth: int


class BacktrackEngine:
    """Manages backtracking with causal dependency awareness.

    When a step fails, the engine finds the best checkpoint to restore,
    invalidates causally-dependent steps via the state manager, and
    records the event for the plan trace.

    Args:
        state_manager: The state manager with checkpoints and dependency graph.
        max_depth: Maximum number of consecutive backtracks before giving up.
    """

    def __init__(
        self,
        state_manager: StateManager,
        max_depth: int = 5,
    ) -> None:
        self._state_manager = state_manager
        self._max_depth = max_depth
        self._current_depth: int = 0
        self._events: list[BacktrackEvent] = []
        logger.debug("BacktrackEngine initialized (max_depth=%d)", max_depth)

    @property
    def events(self) -> list[BacktrackEvent]:
        """Return all backtrack events."""
        return list(self._events)

    @property
    def current_depth(self) -> int:
        """Return the current consecutive backtrack depth."""
        return self._current_depth

    @property
    def max_depth(self) -> int:
        """Return the maximum allowed backtrack depth."""
        return self._max_depth

    def can_backtrack(self) -> bool:
        """Check whether another backtrack is allowed.

        Returns:
            True if depth limit is not exceeded and checkpoints exist.
        """
        has_checkpoints = self._state_manager.get_latest_checkpoint() is not None
        within_limit = self._current_depth < self._max_depth
        return has_checkpoints and within_limit

    def backtrack(self, failed_step_id: str, reason: str) -> BacktrackEvent:
        """Execute a backtrack operation.

        Finds the most recent checkpoint, restores state, invalidates
        downstream steps, and records the event.

        Args:
            failed_step_id: The step that failed.
            reason: Explanation of why backtracking was triggered.

        Returns:
            BacktrackEvent describing what happened.

        Raises:
            BacktrackError: If no checkpoint is available or max depth exceeded.
        """
        if self._current_depth >= self._max_depth:
            raise BacktrackError(
                message=f"Max backtrack depth reached after step '{failed_step_id}'",
                depth=self._current_depth,
                max_depth=self._max_depth,
            )

        checkpoint_id = self.find_best_checkpoint(failed_step_id)
        if checkpoint_id is None:
            raise BacktrackError(
                message="No checkpoint available for backtracking",
                depth=self._current_depth,
                max_depth=self._max_depth,
            )

        invalidated = self._state_manager.invalidate_step(failed_step_id)
        self._state_manager.rollback_to_checkpoint(checkpoint_id)
        self._current_depth += 1

        event = BacktrackEvent(
            trigger_step_id=failed_step_id,
            checkpoint_id=checkpoint_id,
            invalidated_steps=invalidated,
            reason=reason,
            depth=self._current_depth,
        )
        self._events.append(event)

        logger.info(
            "Backtracked from step '%s' to checkpoint '%s' "
            "(depth=%d, invalidated=%d steps)",
            failed_step_id,
            checkpoint_id,
            self._current_depth,
            len(invalidated),
        )
        return event

    def reset_depth(self) -> None:
        """Reset the consecutive backtrack counter.

        Called after a successful step following a backtrack, indicating
        forward progress has resumed.
        """
        if self._current_depth > 0:
            logger.debug("Backtrack depth reset from %d to 0", self._current_depth)
        self._current_depth = 0

    def find_best_checkpoint(self, failed_step_id: str) -> str | None:
        """Find the best checkpoint to rewind to for a given failure.

        Strategy: find the most recent checkpoint that predates any
        step in the failed step's causal chain.

        Args:
            failed_step_id: The step that failed.

        Returns:
            Checkpoint ID, or None if no suitable checkpoint exists.
        """
        checkpoints = self._state_manager.checkpoints
        if not checkpoints:
            return None

        dependent_steps = set(self._state_manager.get_dependents(failed_step_id))

        for checkpoint in reversed(checkpoints):
            if checkpoint.step_index < self._state_manager.step_index:
                cp_step_id = checkpoint.metadata.get("step_id", "")
                if not cp_step_id or cp_step_id not in dependent_steps:
                    return checkpoint.id

        # Fallback: return the most recent checkpoint before current step,
        # even if it is a causal ancestor.
        for checkpoint in reversed(checkpoints):
            if checkpoint.step_index < self._state_manager.step_index:
                return checkpoint.id

        return checkpoints[-1].id if checkpoints else None

    def get_summary(self) -> dict[str, int | list[str]]:
        """Return a summary of all backtracking activity.

        Returns:
            Dict with total backtracks, trigger steps, and depth history.
        """
        return {
            "total_backtracks": len(self._events),
            "trigger_steps": [e.trigger_step_id for e in self._events],
            "max_depth_reached": max((e.depth for e in self._events), default=0),
            "total_invalidated_steps": sum(
                len(e.invalidated_steps) for e in self._events
            ),
        }
