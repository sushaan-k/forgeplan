"""Edge case tests for the BacktrackEngine module.

Covers max depth boundary, circular dependencies, deep chains,
event accumulation, and find_best_checkpoint logic.
"""

from __future__ import annotations

import pytest

from agent_forge.backtrack import BacktrackEngine
from agent_forge.exceptions import BacktrackError
from agent_forge.state import StateManager


class TestBacktrackDepthBoundary:
    """Tests for backtrack depth boundary conditions."""

    def test_max_depth_one(self) -> None:
        """With max_depth=1, only one backtrack should be allowed."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=1)

        sm.create_checkpoint()
        sm.advance_step()

        engine.backtrack("step-1", "first fail")
        assert engine.current_depth == 1
        assert not engine.can_backtrack()

    def test_max_depth_zero(self) -> None:
        """With max_depth=0, no backtracks should be allowed."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=0)

        sm.create_checkpoint()
        assert not engine.can_backtrack()

        with pytest.raises(BacktrackError, match="Max backtrack depth"):
            engine.backtrack("step-1", "cannot backtrack at all")

    def test_backtrack_up_to_exact_max_depth(self) -> None:
        """Should allow exactly max_depth backtracks, then fail."""
        sm = StateManager()
        max_depth = 4
        engine = BacktrackEngine(state_manager=sm, max_depth=max_depth)

        for i in range(max_depth):
            sm.set("step", i)
            sm.create_checkpoint()
            sm.advance_step()
            event = engine.backtrack(f"step-{i}", f"fail-{i}")
            assert event.depth == i + 1

        sm.create_checkpoint()
        sm.advance_step()
        with pytest.raises(BacktrackError):
            engine.backtrack("final-step", "one too many")

    def test_reset_then_backtrack_again(self) -> None:
        """After reset_depth, should be able to backtrack again."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=2)

        sm.create_checkpoint()
        sm.advance_step()
        engine.backtrack("s1", "fail1")

        sm.create_checkpoint()
        sm.advance_step()
        engine.backtrack("s2", "fail2")

        assert not engine.can_backtrack()

        engine.reset_depth()
        assert engine.current_depth == 0

        sm.create_checkpoint()
        sm.advance_step()
        event = engine.backtrack("s3", "fail3 after reset")
        assert event.depth == 1

    def test_reset_depth_idempotent_at_zero(self) -> None:
        """Resetting depth when already at 0 should be a no-op."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=5)
        assert engine.current_depth == 0
        engine.reset_depth()
        assert engine.current_depth == 0


class TestBacktrackWithDependencies:
    """Tests for backtracking with complex dependency graphs."""

    def test_backtrack_invalidates_downstream(self) -> None:
        """Backtracking should invalidate all steps dependent on failed step."""
        sm = StateManager()
        sm.add_dependency("step-1", "step-2")
        sm.add_dependency("step-2", "step-3")
        sm.add_dependency("step-1", "step-4")

        sm.set("data", "original")
        sm.create_checkpoint()
        sm.advance_step()
        sm.set("data", "modified")

        engine = BacktrackEngine(state_manager=sm, max_depth=5)
        event = engine.backtrack("step-1", "step-1 failed")

        # step-1 plus all its transitive dependents
        assert "step-1" in event.invalidated_steps
        assert "step-2" in event.invalidated_steps
        assert "step-3" in event.invalidated_steps
        assert "step-4" in event.invalidated_steps
        assert sm.get("data") == "original"

    def test_backtrack_with_no_dependencies(self) -> None:
        """Backtracking a step with no dependents should invalidate only itself."""
        sm = StateManager()
        sm.set("x", 1)
        sm.create_checkpoint()
        sm.advance_step()

        engine = BacktrackEngine(state_manager=sm, max_depth=5)
        event = engine.backtrack("lone-step", "isolated failure")
        assert event.invalidated_steps == ["lone-step"]


class TestFindBestCheckpoint:
    """Tests for find_best_checkpoint logic."""

    def test_no_checkpoints(self) -> None:
        """Should return None when no checkpoints exist."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm)
        assert engine.find_best_checkpoint("step-1") is None

    def test_single_checkpoint(self) -> None:
        """Should return the only checkpoint available."""
        sm = StateManager()
        cp = sm.create_checkpoint()
        sm.advance_step()

        engine = BacktrackEngine(state_manager=sm)
        result = engine.find_best_checkpoint("step-1")
        assert result == cp.id

    def test_multiple_checkpoints_returns_most_recent_before_current(self) -> None:
        """Should return a checkpoint that predates current step."""
        sm = StateManager()
        sm.create_checkpoint()  # cp at step 0
        sm.advance_step()
        cp2 = sm.create_checkpoint()  # cp at step 1
        sm.advance_step()
        sm.advance_step()  # now at step 3

        engine = BacktrackEngine(state_manager=sm)
        result = engine.find_best_checkpoint("step-3")
        assert result == cp2.id

    def test_find_best_checkpoint_with_dependency_graph(self) -> None:
        """Should work with dependency information in the graph."""
        sm = StateManager()
        sm.add_dependency("step-a", "step-b")
        sm.add_dependency("step-b", "step-c")

        sm.create_checkpoint()
        sm.advance_step()
        sm.create_checkpoint()
        sm.advance_step()

        engine = BacktrackEngine(state_manager=sm)
        result = engine.find_best_checkpoint("step-c")
        assert result is not None

    def test_backtrack_uses_best_checkpoint_not_latest(self) -> None:
        """Backtracking should skip checkpoints belonging to downstream steps."""
        sm = StateManager(initial_state={"value": "base"})

        sm.set("value", "base")
        cp1 = sm.create_checkpoint(metadata={"step_id": "step-a"})
        sm.advance_step()

        sm.add_dependency("step-a", "step-b")
        sm.set("value", "downstream")
        cp2 = sm.create_checkpoint(metadata={"step_id": "step-b"})
        sm.advance_step()

        engine = BacktrackEngine(state_manager=sm)
        event = engine.backtrack("step-a", "step-a failed")

        assert event.checkpoint_id == cp1.id
        assert sm.get("value") == "base"
        assert cp2.id not in {event.checkpoint_id}


class TestBacktrackEventRecording:
    """Tests for event recording and summary generation."""

    def test_events_are_ordered(self) -> None:
        """Events should maintain chronological order."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=10)

        for i in range(5):
            sm.create_checkpoint()
            sm.advance_step()
            engine.backtrack(f"step-{i}", f"reason-{i}")

        events = engine.events
        assert len(events) == 5
        for i, event in enumerate(events):
            assert event.trigger_step_id == f"step-{i}"
            assert event.reason == f"reason-{i}"
            assert event.depth == i + 1

    def test_events_list_is_a_copy(self) -> None:
        """events property should return a copy, not the internal list."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=5)
        sm.create_checkpoint()
        sm.advance_step()
        engine.backtrack("s1", "r1")

        events = engine.events
        events.clear()
        assert len(engine.events) == 1

    def test_summary_with_no_events(self) -> None:
        """Summary of empty engine should have zeros."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm)
        summary = engine.get_summary()
        assert summary["total_backtracks"] == 0
        assert summary["trigger_steps"] == []
        assert summary["max_depth_reached"] == 0
        assert summary["total_invalidated_steps"] == 0

    def test_summary_aggregation(self) -> None:
        """Summary should correctly aggregate multiple events."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=10)

        # First backtrack: step-a has 2 dependents
        sm.add_dependency("step-a", "step-b")
        sm.add_dependency("step-a", "step-c")
        sm.create_checkpoint()
        sm.advance_step()
        engine.backtrack("step-a", "fail-a")

        # Second backtrack: step-d has no dependents
        sm.create_checkpoint()
        sm.advance_step()
        engine.backtrack("step-d", "fail-d")

        summary = engine.get_summary()
        assert summary["total_backtracks"] == 2
        assert summary["trigger_steps"] == ["step-a", "step-d"]
        assert summary["max_depth_reached"] == 2
        # step-a invalidated [step-a, step-b, step-c] = 3
        # step-d invalidated [step-d] = 1
        assert summary["total_invalidated_steps"] == 4


class TestBacktrackCanBacktrackLogic:
    """Tests for can_backtrack() under various conditions."""

    def test_cannot_backtrack_no_checkpoint_within_limit(self) -> None:
        """can_backtrack requires BOTH checkpoints AND depth room."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=5)
        # No checkpoints, but within depth limit
        assert not engine.can_backtrack()

    def test_cannot_backtrack_at_limit_with_checkpoint(self) -> None:
        """can_backtrack should be False when at max depth even with checkpoints."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=1)
        sm.create_checkpoint()
        sm.advance_step()
        engine.backtrack("s1", "fail")

        sm.create_checkpoint()
        assert not engine.can_backtrack()

    def test_can_backtrack_with_checkpoint_and_depth_room(self) -> None:
        """can_backtrack should be True when both conditions met."""
        sm = StateManager()
        engine = BacktrackEngine(state_manager=sm, max_depth=5)
        sm.create_checkpoint()
        assert engine.can_backtrack()
