"""Edge case tests for the StateManager module.

Covers boundary conditions, complex dependency graphs, checkpoint
chains, rollback under stress, and state isolation guarantees.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from agent_forge.exceptions import StateError
from agent_forge.state import StateManager


class TestStateManagerBoundaryConditions:
    """Boundary conditions for state operations."""

    def test_set_none_value(self) -> None:
        """Setting a key to None should store None, not remove the key."""
        sm = StateManager()
        sm.set("key", None)
        assert sm.get("key") is None
        assert "key" in sm.current_state

    def test_set_empty_string_key(self) -> None:
        """Empty string is a valid key."""
        sm = StateManager()
        sm.set("", "empty_key_value")
        assert sm.get("") == "empty_key_value"

    def test_update_with_empty_dict(self) -> None:
        """Batch update with empty dict should be a no-op."""
        sm = StateManager(initial_state={"existing": 1})
        sm.update({})
        assert sm.get("existing") == 1

    def test_update_overwrites_existing_keys(self) -> None:
        """Batch update should overwrite existing keys."""
        sm = StateManager(initial_state={"a": 1, "b": 2})
        sm.update({"a": 99, "c": 3})
        assert sm.get("a") == 99
        assert sm.get("b") == 2
        assert sm.get("c") == 3

    def test_nested_mutable_state_isolation(self) -> None:
        """Deep mutations on retrieved state must not leak back."""
        sm = StateManager(initial_state={"nested": {"inner": [1, 2, 3]}})
        retrieved = sm.current_state
        retrieved["nested"]["inner"].append(4)
        assert sm.get("nested") == {"inner": [1, 2, 3]}

    def test_set_complex_values(self) -> None:
        """State should handle complex nested data structures."""
        sm = StateManager()
        complex_val: dict[str, Any] = {
            "list": [1, [2, 3], {"nested": True}],
            "tuple_as_list": [1, 2, 3],
            "none": None,
        }
        sm.set("complex", complex_val)
        assert sm.get("complex") == complex_val

    def test_advance_step_many_times(self) -> None:
        """Step index should increment reliably over many advances."""
        sm = StateManager()
        for i in range(100):
            new_idx = sm.advance_step()
            assert new_idx == i + 1
        assert sm.step_index == 100

    def test_initial_state_not_mutated(self) -> None:
        """The dict passed as initial_state should not be mutated."""
        original = {"key": "value"}
        original_copy = copy.deepcopy(original)
        sm = StateManager(initial_state=original)
        sm.set("key", "changed")
        sm.set("new_key", "new_value")
        assert original == original_copy


class TestCheckpointEdgeCases:
    """Edge cases for checkpoint and rollback operations."""

    def test_checkpoint_with_empty_state(self) -> None:
        """Checkpointing empty state should work."""
        sm = StateManager()
        cp = sm.create_checkpoint()
        assert cp.state_snapshot == {}
        assert cp.step_index == 0

    def test_checkpoint_snapshot_is_independent(self) -> None:
        """Checkpoint snapshot should be independent of later state changes."""
        sm = StateManager()
        sm.set("key", "original")
        cp = sm.create_checkpoint()
        sm.set("key", "modified")
        assert cp.state_snapshot["key"] == "original"

    def test_rollback_to_first_of_many_checkpoints(self) -> None:
        """Rolling back to the very first checkpoint should trim all later ones."""
        sm = StateManager()
        sm.set("counter", 0)
        cp0 = sm.create_checkpoint()

        for i in range(1, 10):
            sm.set("counter", i)
            sm.advance_step()
            sm.create_checkpoint()

        sm.rollback_to_checkpoint(cp0.id)
        assert sm.get("counter") == 0
        assert sm.step_index == 0
        assert len(sm.checkpoints) == 1

    def test_rollback_then_create_new_checkpoint(self) -> None:
        """After rollback, new checkpoints should work properly."""
        sm = StateManager()
        sm.set("phase", "alpha")
        cp1 = sm.create_checkpoint()

        sm.set("phase", "beta")
        sm.advance_step()
        sm.create_checkpoint()

        sm.rollback_to_checkpoint(cp1.id)
        assert sm.get("phase") == "alpha"

        sm.set("phase", "gamma")
        sm.advance_step()
        cp3 = sm.create_checkpoint()
        assert cp3.state_snapshot["phase"] == "gamma"
        assert len(sm.checkpoints) == 2

    def test_rollback_twice_to_same_checkpoint(self) -> None:
        """Rolling back to the same checkpoint twice should be idempotent."""
        sm = StateManager()
        sm.set("val", 10)
        cp = sm.create_checkpoint()

        sm.set("val", 20)
        sm.advance_step()
        sm.rollback_to_checkpoint(cp.id)
        assert sm.get("val") == 10

        # Rolling back again to same checkpoint (it's still the latest)
        sm.set("val", 30)
        sm.advance_step()
        sm.rollback_to_checkpoint(cp.id)
        assert sm.get("val") == 10

    def test_rollback_nonexistent_checkpoint_error(self) -> None:
        """Rolling back to a nonexistent checkpoint should raise StateError."""
        sm = StateManager()
        sm.create_checkpoint()
        with pytest.raises(StateError, match="not found"):
            sm.rollback_to_checkpoint("does-not-exist-id")

    def test_checkpoint_metadata_preserved(self) -> None:
        """Checkpoint metadata should be preserved and retrievable."""
        sm = StateManager()
        meta = {"reason": "pre_deploy", "author": "test"}
        cp = sm.create_checkpoint(metadata=meta)
        assert cp.metadata == meta

    def test_multiple_rollbacks_forward_and_back(self) -> None:
        """State should be consistent through multiple rollback-advance cycles."""
        sm = StateManager()
        sm.set("x", 0)
        cp0 = sm.create_checkpoint()

        sm.set("x", 1)
        sm.advance_step()
        cp1 = sm.create_checkpoint()

        sm.set("x", 2)
        sm.advance_step()

        # Rollback to cp1, advance, rollback to cp0
        sm.rollback_to_checkpoint(cp1.id)
        assert sm.get("x") == 1
        assert sm.step_index == 1

        sm.set("x", 99)
        sm.advance_step()
        sm.rollback_to_checkpoint(cp0.id)
        assert sm.get("x") == 0
        assert sm.step_index == 0


class TestDependencyGraphEdgeCases:
    """Edge cases for causal dependency graph operations."""

    def test_self_loop_dependency(self) -> None:
        """Adding a self-loop should work without errors."""
        sm = StateManager()
        sm.add_dependency("a", "a")
        # networkx allows self-loops; get_dependents should not loop infinitely
        deps = sm.get_dependents("a")
        # "a" depends on itself -- descendants includes itself via self-loop
        assert "a" in deps or deps == []

    def test_diamond_dependency(self) -> None:
        """Diamond-shaped dependency graph should resolve correctly."""
        sm = StateManager()
        #     a
        #    / \
        #   b   c
        #    \ /
        #     d
        sm.add_dependency("a", "b")
        sm.add_dependency("a", "c")
        sm.add_dependency("b", "d")
        sm.add_dependency("c", "d")

        deps_of_a = sm.get_dependents("a")
        assert set(deps_of_a) == {"b", "c", "d"}

    def test_deep_chain_dependency(self) -> None:
        """Long dependency chain should resolve correctly."""
        sm = StateManager()
        chain = [f"step-{i}" for i in range(20)]
        for i in range(len(chain) - 1):
            sm.add_dependency(chain[i], chain[i + 1])

        deps = sm.get_dependents("step-0")
        assert len(deps) == 19
        assert "step-19" in deps

    def test_invalidate_leaf_node(self) -> None:
        """Invalidating a leaf node with no dependents should invalidate only itself."""
        sm = StateManager()
        sm.add_dependency("a", "b")
        sm.add_dependency("b", "c")

        invalidated = sm.invalidate_step("c")
        assert invalidated == ["c"]

    def test_invalidate_middle_node(self) -> None:
        """Invalidating a middle node should cascade to all descendants."""
        sm = StateManager()
        sm.add_dependency("a", "b")
        sm.add_dependency("b", "c")
        sm.add_dependency("b", "d")

        invalidated = sm.invalidate_step("b")
        assert set(invalidated) == {"b", "c", "d"}

    def test_invalidate_removes_from_graph(self) -> None:
        """After invalidation, the nodes should be removed from the graph."""
        sm = StateManager()
        sm.add_dependency("a", "b")
        sm.add_dependency("b", "c")

        sm.invalidate_step("b")

        # b and c are removed, a should have no dependents
        deps = sm.get_dependents("a")
        assert deps == []

    def test_dependency_graph_is_copy(self) -> None:
        """dependency_graph property should return a copy, not the internal graph."""
        sm = StateManager()
        sm.add_dependency("a", "b")
        graph = sm.dependency_graph
        graph.remove_edge("a", "b")
        # Internal graph should be unaffected
        internal_graph = sm.dependency_graph
        assert internal_graph.has_edge("a", "b")

    def test_invalidate_nonexistent_step(self) -> None:
        """Invalidating a step not in the graph should return just that step."""
        sm = StateManager()
        sm.add_dependency("a", "b")
        invalidated = sm.invalidate_step("nonexistent")
        assert invalidated == ["nonexistent"]

    def test_complex_dag_invalidation(self) -> None:
        """Complex DAG with multiple paths should invalidate all reachable nodes."""
        sm = StateManager()
        # Build a complex DAG:
        #   a -> b -> d -> f
        #   a -> c -> e -> f
        #   b -> e
        sm.add_dependency("a", "b")
        sm.add_dependency("a", "c")
        sm.add_dependency("b", "d")
        sm.add_dependency("b", "e")
        sm.add_dependency("c", "e")
        sm.add_dependency("d", "f")
        sm.add_dependency("e", "f")

        invalidated = sm.invalidate_step("a")
        assert set(invalidated) == {"a", "b", "c", "d", "e", "f"}

    def test_reset_clears_dependency_graph(self) -> None:
        """Reset should clear the dependency graph entirely."""
        sm = StateManager()
        sm.add_dependency("a", "b")
        sm.add_dependency("b", "c")
        sm.reset()
        graph = sm.dependency_graph
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0


class TestStateManagerReset:
    """Tests for full state reset."""

    def test_reset_restores_clean_state(self) -> None:
        """After reset, all state should be as if freshly constructed."""
        sm = StateManager(initial_state={"key": "value"})
        sm.set("extra", "data")
        sm.advance_step()
        sm.create_checkpoint()
        sm.add_dependency("a", "b")

        sm.reset()
        assert sm.current_state == {}
        assert sm.step_index == 0
        assert sm.checkpoints == []
        assert len(sm.dependency_graph.nodes) == 0

    def test_usable_after_reset(self) -> None:
        """State manager should be fully usable after reset."""
        sm = StateManager()
        sm.set("before", True)
        sm.create_checkpoint()
        sm.advance_step()

        sm.reset()

        sm.set("after", True)
        cp = sm.create_checkpoint()
        sm.advance_step()
        sm.rollback_to_checkpoint(cp.id)
        assert sm.get("after") is True
        assert sm.get("before") is None
