"""State manager with checkpointing, rollback, and causal dependency tracking.

The StateManager maintains the world state as a versioned dictionary,
builds a causal dependency graph of actions, and supports checkpoint /
rollback so the backtracking engine can rewind to any known-good point.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from agent_forge.exceptions import StateError

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A snapshot of world state at a given point in time.

    Attributes:
        id: Unique checkpoint identifier.
        step_index: The step number when this checkpoint was taken.
        state_snapshot: Deep copy of the world state at checkpoint time.
        metadata: Optional context about why the checkpoint was created.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    step_index: int = 0
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


class StateManager:
    """Manages world state, checkpoints, and causal dependencies.

    The state manager tracks:
    - Current world state as a key-value store.
    - Checkpoints for rollback support.
    - A directed acyclic graph of causal dependencies between actions.

    Args:
        initial_state: Starting world state. Defaults to empty dict.
    """

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = dict(initial_state or {})
        self._checkpoints: list[Checkpoint] = []
        self._dependency_graph: nx.DiGraph = nx.DiGraph()
        self._step_index: int = 0
        self._history: list[dict[str, Any]] = [copy.deepcopy(self._state)]
        logger.debug("StateManager initialized with %d keys", len(self._state))

    @property
    def current_state(self) -> dict[str, Any]:
        """Return the current world state (read-only copy)."""
        return copy.deepcopy(self._state)

    @property
    def step_index(self) -> int:
        """Return the current step index."""
        return self._step_index

    @property
    def checkpoints(self) -> list[Checkpoint]:
        """Return all stored checkpoints."""
        return list(self._checkpoints)

    @property
    def dependency_graph(self) -> nx.DiGraph:
        """Return the causal dependency graph."""
        return self._dependency_graph.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the world state.

        Args:
            key: State key to look up.
            default: Value returned if key is missing.

        Returns:
            The state value, or `default` if not found.
        """
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the world state.

        Args:
            key: State key to update.
            value: New value.
        """
        self._state[key] = value
        logger.debug("State updated: %s = %s", key, repr(value)[:80])

    def update(self, updates: dict[str, Any]) -> None:
        """Batch-update multiple state keys.

        Args:
            updates: Dictionary of key-value pairs to merge into state.
        """
        self._state.update(updates)
        logger.debug("State batch-updated with %d keys", len(updates))

    def advance_step(self) -> int:
        """Advance the step counter and record state history.

        Returns:
            The new step index.
        """
        self._step_index += 1
        self._history.append(copy.deepcopy(self._state))
        return self._step_index

    def create_checkpoint(self, metadata: dict[str, str] | None = None) -> Checkpoint:
        """Create a checkpoint of the current world state.

        Args:
            metadata: Optional context about the checkpoint.

        Returns:
            The newly created Checkpoint.
        """
        checkpoint = Checkpoint(
            step_index=self._step_index,
            state_snapshot=copy.deepcopy(self._state),
            metadata=metadata or {},
        )
        self._checkpoints.append(checkpoint)
        logger.info(
            "Checkpoint created: id=%s step=%d", checkpoint.id, checkpoint.step_index
        )
        return checkpoint

    def rollback_to_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Restore world state from a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to restore.

        Returns:
            The restored world state.

        Raises:
            StateError: If the checkpoint ID is not found.
        """
        target: Checkpoint | None = None
        target_idx: int = -1
        for idx, cp in enumerate(self._checkpoints):
            if cp.id == checkpoint_id:
                target = cp
                target_idx = idx
                break

        if target is None:
            raise StateError(f"Checkpoint '{checkpoint_id}' not found")

        self._state = copy.deepcopy(target.state_snapshot)
        self._step_index = target.step_index
        # Trim history to checkpoint point
        self._history = self._history[: target.step_index + 1]
        # Remove checkpoints that came after this one
        self._checkpoints = self._checkpoints[: target_idx + 1]
        logger.info(
            "Rolled back to checkpoint %s (step %d)",
            checkpoint_id,
            target.step_index,
        )
        return copy.deepcopy(self._state)

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Return the most recent checkpoint, or None if none exist."""
        return self._checkpoints[-1] if self._checkpoints else None

    def add_dependency(self, from_step: str, to_step: str) -> None:
        """Record a causal dependency between two steps.

        Args:
            from_step: The step that must complete first.
            to_step: The step that depends on from_step.
        """
        self._dependency_graph.add_edge(from_step, to_step)
        logger.debug("Dependency added: %s -> %s", from_step, to_step)

    def get_dependents(self, step_id: str) -> list[str]:
        """Find all steps that transitively depend on a given step.

        Args:
            step_id: The step whose dependents to find.

        Returns:
            List of step IDs that depend on the given step.
        """
        if step_id not in self._dependency_graph:
            return []
        return list(nx.descendants(self._dependency_graph, step_id))

    def invalidate_step(self, step_id: str) -> list[str]:
        """Invalidate a step and all its transitive dependents.

        Args:
            step_id: The step that failed or was invalidated.

        Returns:
            List of all invalidated step IDs (including the given step).
        """
        dependents = self.get_dependents(step_id)
        all_invalidated = [step_id, *dependents]
        for sid in all_invalidated:
            if sid in self._dependency_graph:
                self._dependency_graph.remove_node(sid)
        logger.info("Invalidated step %s and %d dependents", step_id, len(dependents))
        return all_invalidated

    def reset(self) -> None:
        """Reset state manager to initial conditions."""
        self._state.clear()
        self._checkpoints.clear()
        self._dependency_graph.clear()
        self._step_index = 0
        self._history = [{}]
        logger.info("StateManager reset")
