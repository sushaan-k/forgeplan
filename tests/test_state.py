"""Tests for the StateManager module."""

from __future__ import annotations

import pytest

from agent_forge.exceptions import StateError
from agent_forge.state import Checkpoint, StateManager


class TestStateManager:
    """Tests for state management operations."""

    def test_initial_state_empty(self, state_manager: StateManager) -> None:
        """State should start empty by default."""
        assert state_manager.current_state == {}
        assert state_manager.step_index == 0

    def test_initial_state_populated(self) -> None:
        """State should accept initial values."""
        sm = StateManager(initial_state={"key": "value"})
        assert sm.get("key") == "value"

    def test_get_set(self, state_manager: StateManager) -> None:
        """Should set and get values."""
        state_manager.set("name", "test")
        assert state_manager.get("name") == "test"

    def test_get_default(self, state_manager: StateManager) -> None:
        """Should return default for missing keys."""
        assert state_manager.get("missing") is None
        assert state_manager.get("missing", 42) == 42

    def test_batch_update(self, state_manager: StateManager) -> None:
        """Should batch-update multiple keys."""
        state_manager.update({"a": 1, "b": 2, "c": 3})
        assert state_manager.get("a") == 1
        assert state_manager.get("b") == 2
        assert state_manager.get("c") == 3

    def test_advance_step(self, state_manager: StateManager) -> None:
        """Should increment step counter."""
        assert state_manager.step_index == 0
        new_idx = state_manager.advance_step()
        assert new_idx == 1
        assert state_manager.step_index == 1

    def test_current_state_is_copy(self, state_manager: StateManager) -> None:
        """current_state should return a copy, not a reference."""
        state_manager.set("key", "original")
        copy = state_manager.current_state
        copy["key"] = "modified"
        assert state_manager.get("key") == "original"


class TestCheckpoints:
    """Tests for checkpoint and rollback operations."""

    def test_create_checkpoint(self, state_manager: StateManager) -> None:
        """Should create a checkpoint of current state."""
        state_manager.set("x", 10)
        cp = state_manager.create_checkpoint()
        assert isinstance(cp, Checkpoint)
        assert cp.state_snapshot == {"x": 10}
        assert cp.step_index == 0

    def test_rollback_to_checkpoint(self, state_manager: StateManager) -> None:
        """Should restore state from checkpoint."""
        state_manager.set("x", 10)
        cp = state_manager.create_checkpoint()

        state_manager.set("x", 99)
        state_manager.set("y", 42)
        state_manager.advance_step()

        restored = state_manager.rollback_to_checkpoint(cp.id)
        assert restored == {"x": 10}
        assert state_manager.get("y") is None
        assert state_manager.step_index == 0

    def test_rollback_invalid_checkpoint(self, state_manager: StateManager) -> None:
        """Should raise on invalid checkpoint ID."""
        with pytest.raises(StateError):
            state_manager.rollback_to_checkpoint("nonexistent")

    def test_latest_checkpoint(self, state_manager: StateManager) -> None:
        """Should return the most recent checkpoint."""
        assert state_manager.get_latest_checkpoint() is None

        cp1 = state_manager.create_checkpoint()
        assert state_manager.get_latest_checkpoint() == cp1

        cp2 = state_manager.create_checkpoint()
        assert state_manager.get_latest_checkpoint() == cp2

    def test_multiple_checkpoints_rollback(self, state_manager: StateManager) -> None:
        """Should handle rolling back with multiple checkpoints."""
        state_manager.set("phase", "start")
        cp1 = state_manager.create_checkpoint()

        state_manager.set("phase", "middle")
        state_manager.advance_step()
        state_manager.create_checkpoint()

        state_manager.set("phase", "end")
        state_manager.advance_step()

        state_manager.rollback_to_checkpoint(cp1.id)
        assert state_manager.get("phase") == "start"
        assert len(state_manager.checkpoints) == 1


class TestDependencyGraph:
    """Tests for causal dependency tracking."""

    def test_add_dependency(self, state_manager: StateManager) -> None:
        """Should add edges to the dependency graph."""
        state_manager.add_dependency("step-1", "step-2")
        graph = state_manager.dependency_graph
        assert graph.has_edge("step-1", "step-2")

    def test_get_dependents(self, state_manager: StateManager) -> None:
        """Should find transitive dependents."""
        state_manager.add_dependency("a", "b")
        state_manager.add_dependency("b", "c")
        state_manager.add_dependency("b", "d")

        deps = state_manager.get_dependents("a")
        assert set(deps) == {"b", "c", "d"}

    def test_get_dependents_no_deps(self, state_manager: StateManager) -> None:
        """Should return empty list for nodes with no dependents."""
        assert state_manager.get_dependents("nonexistent") == []

    def test_invalidate_step(self, state_manager: StateManager) -> None:
        """Should invalidate a step and its dependents."""
        state_manager.add_dependency("a", "b")
        state_manager.add_dependency("b", "c")

        invalidated = state_manager.invalidate_step("a")
        assert "a" in invalidated
        assert "b" in invalidated
        assert "c" in invalidated

    def test_reset(self, state_manager: StateManager) -> None:
        """Should reset all state."""
        state_manager.set("key", "value")
        state_manager.create_checkpoint()
        state_manager.add_dependency("a", "b")
        state_manager.advance_step()

        state_manager.reset()
        assert state_manager.current_state == {}
        assert state_manager.step_index == 0
        assert state_manager.checkpoints == []
