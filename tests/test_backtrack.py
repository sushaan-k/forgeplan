"""Tests for the BacktrackEngine module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_forge.backtrack import BacktrackEngine, BacktrackEvent
from agent_forge.exceptions import BacktrackError

if TYPE_CHECKING:
    from agent_forge.state import StateManager


class TestBacktrackEngine:
    """Tests for backtracking operations."""

    def test_initial_state(self, backtrack_engine: BacktrackEngine) -> None:
        """BacktrackEngine should start with zero depth."""
        assert backtrack_engine.current_depth == 0
        assert backtrack_engine.events == []
        assert backtrack_engine.max_depth == 3

    def test_can_backtrack_no_checkpoints(self, state_manager: StateManager) -> None:
        """Cannot backtrack without any checkpoints."""
        engine = BacktrackEngine(state_manager=state_manager)
        assert not engine.can_backtrack()

    def test_can_backtrack_with_checkpoint(self, state_manager: StateManager) -> None:
        """Can backtrack when checkpoints exist."""
        state_manager.create_checkpoint()
        engine = BacktrackEngine(state_manager=state_manager)
        assert engine.can_backtrack()

    def test_backtrack_restores_state(self, state_manager: StateManager) -> None:
        """Backtracking should restore state to checkpoint."""
        state_manager.set("value", 10)
        state_manager.create_checkpoint()

        state_manager.set("value", 99)
        state_manager.advance_step()

        engine = BacktrackEngine(state_manager=state_manager)
        event = engine.backtrack("failed-step", "test failure")

        assert state_manager.get("value") == 10
        assert isinstance(event, BacktrackEvent)
        assert event.trigger_step_id == "failed-step"
        assert event.reason == "test failure"
        assert event.depth == 1

    def test_backtrack_depth_tracking(self, state_manager: StateManager) -> None:
        """Depth should increment with each backtrack."""
        engine = BacktrackEngine(state_manager=state_manager, max_depth=5)

        for i in range(3):
            state_manager.set("step", i)
            state_manager.create_checkpoint()
            state_manager.advance_step()

        engine.backtrack(f"step-{0}", "fail 1")
        assert engine.current_depth == 1

    def test_backtrack_exceeds_max_depth(self, state_manager: StateManager) -> None:
        """Should raise BacktrackError when max depth exceeded."""
        engine = BacktrackEngine(state_manager=state_manager, max_depth=1)

        state_manager.create_checkpoint()
        state_manager.advance_step()
        engine.backtrack("step-1", "first fail")

        state_manager.create_checkpoint()
        state_manager.advance_step()

        with pytest.raises(BacktrackError) as exc_info:
            engine.backtrack("step-2", "second fail")
        assert "Max backtrack depth" in str(exc_info.value)

    def test_backtrack_no_checkpoint_raises(self, state_manager: StateManager) -> None:
        """Should raise BacktrackError when no checkpoint exists."""
        engine = BacktrackEngine(state_manager=state_manager)
        with pytest.raises(BacktrackError) as exc_info:
            engine.backtrack("step-1", "no checkpoint")
        assert "No checkpoint available" in str(exc_info.value)

    def test_reset_depth(self, state_manager: StateManager) -> None:
        """reset_depth should zero the counter."""
        engine = BacktrackEngine(state_manager=state_manager, max_depth=5)
        state_manager.create_checkpoint()
        state_manager.advance_step()

        engine.backtrack("step-1", "test")
        assert engine.current_depth == 1

        engine.reset_depth()
        assert engine.current_depth == 0

    def test_events_recorded(self, state_manager: StateManager) -> None:
        """All backtrack events should be recorded."""
        engine = BacktrackEngine(state_manager=state_manager, max_depth=5)

        state_manager.create_checkpoint()
        state_manager.advance_step()
        engine.backtrack("s1", "reason1")

        state_manager.create_checkpoint()
        state_manager.advance_step()
        engine.backtrack("s2", "reason2")

        events = engine.events
        assert len(events) == 2
        assert events[0].trigger_step_id == "s1"
        assert events[1].trigger_step_id == "s2"

    def test_get_summary(self, state_manager: StateManager) -> None:
        """Summary should aggregate backtrack statistics."""
        engine = BacktrackEngine(state_manager=state_manager, max_depth=5)

        state_manager.create_checkpoint()
        state_manager.advance_step()
        engine.backtrack("s1", "r1")

        summary = engine.get_summary()
        assert summary["total_backtracks"] == 1
        assert "s1" in summary["trigger_steps"]
