"""Edge case tests for the Goal module.

Covers validation boundaries, status transitions, metadata handling,
and goal model serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_forge.goal import Goal, GoalStatus


class TestGoalValidation:
    """Validation edge cases for Goal."""

    def test_max_steps_exactly_one(self) -> None:
        """max_steps=1 should be valid (minimum allowed)."""
        goal = Goal(description="One step", max_steps=1)
        assert goal.max_steps == 1

    def test_max_steps_negative_rejected(self) -> None:
        """Negative max_steps should be rejected."""
        with pytest.raises(ValidationError):
            Goal(description="bad", max_steps=-5)

    def test_max_steps_very_large(self) -> None:
        """Very large max_steps should be accepted."""
        goal = Goal(description="long", max_steps=1_000_000)
        assert goal.max_steps == 1_000_000

    def test_priority_zero(self) -> None:
        """Priority of 0 should be valid."""
        goal = Goal(description="low priority", priority=0)
        assert goal.priority == 0

    def test_priority_negative_rejected(self) -> None:
        """Negative priority should be rejected."""
        with pytest.raises(ValidationError):
            Goal(description="bad", priority=-1)

    def test_priority_high_value(self) -> None:
        """High priority value should be accepted."""
        goal = Goal(description="urgent", priority=999)
        assert goal.priority == 999


class TestGoalStatusTransitions:
    """Tests for goal status transition sequences."""

    def test_pending_to_completed_directly(self) -> None:
        """Should allow transition from PENDING to COMPLETED directly."""
        goal = Goal(description="quick")
        goal.mark_completed()
        assert goal.status == GoalStatus.COMPLETED

    def test_pending_to_failed_directly(self) -> None:
        """Should allow transition from PENDING to FAILED directly."""
        goal = Goal(description="instant fail")
        goal.mark_failed()
        assert goal.status == GoalStatus.FAILED

    def test_completed_to_failed(self) -> None:
        """Transitions are unconstrained -- completed can go to failed."""
        goal = Goal(description="test")
        goal.mark_completed()
        goal.mark_failed()
        assert goal.status == GoalStatus.FAILED

    def test_cancelled_to_in_progress(self) -> None:
        """Even cancelled goals can be restarted (transitions are unconstrained)."""
        goal = Goal(description="test")
        goal.mark_cancelled()
        goal.mark_in_progress()
        assert goal.status == GoalStatus.IN_PROGRESS

    def test_all_status_values(self) -> None:
        """All GoalStatus enum values should be valid."""
        expected = {"pending", "in_progress", "completed", "failed", "cancelled"}
        actual = {s.value for s in GoalStatus}
        assert actual == expected


class TestGoalMetadata:
    """Tests for goal metadata handling."""

    def test_empty_metadata(self) -> None:
        """Empty metadata should be an empty dict."""
        goal = Goal(description="test")
        assert goal.metadata == {}

    def test_metadata_with_values(self) -> None:
        """Metadata should store key-value pairs."""
        goal = Goal(
            description="test",
            metadata={"env": "production", "user": "admin"},
        )
        assert goal.metadata["env"] == "production"
        assert goal.metadata["user"] == "admin"

    def test_metadata_many_keys(self) -> None:
        """Metadata should handle many keys."""
        meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        goal = Goal(description="test", metadata=meta)
        assert len(goal.metadata) == 100


class TestGoalSuccessCriteria:
    """Tests for success criteria handling."""

    def test_empty_success_criteria(self) -> None:
        """Empty success criteria should be valid."""
        goal = Goal(description="test", success_criteria=[])
        assert goal.success_criteria == []

    def test_single_criterion(self) -> None:
        """Single success criterion should work."""
        goal = Goal(description="test", success_criteria=["task done"])
        assert goal.success_criteria == ["task done"]

    def test_many_criteria(self) -> None:
        """Many success criteria should work."""
        criteria = [f"criterion_{i}" for i in range(50)]
        goal = Goal(description="test", success_criteria=criteria)
        assert len(goal.success_criteria) == 50


class TestGoalInvariants:
    """Tests for invariant handling."""

    def test_empty_invariants(self) -> None:
        """Empty invariants should be valid."""
        goal = Goal(description="test", invariants=[])
        assert goal.invariants == []

    def test_many_invariants(self) -> None:
        """Many invariants should work."""
        invariants = [f"rule_{i}" for i in range(50)]
        goal = Goal(description="test", invariants=invariants)
        assert len(goal.invariants) == 50


class TestGoalSerialization:
    """Tests for Goal model serialization."""

    def test_goal_to_dict(self) -> None:
        """Goal should be serializable to dict via model_dump."""
        goal = Goal(
            description="test",
            success_criteria=["done"],
            max_steps=10,
            priority=5,
        )
        data = goal.model_dump()
        assert data["description"] == "test"
        assert data["max_steps"] == 10
        assert data["priority"] == 5
        assert "id" in data

    def test_goal_from_dict(self) -> None:
        """Goal should be deserializable from dict."""
        data = {
            "id": "test-id-1234",
            "description": "from dict",
            "success_criteria": ["c1"],
            "max_steps": 50,
            "invariants": [],
            "priority": 2,
            "status": "pending",
            "metadata": {},
        }
        goal = Goal(**data)
        assert goal.id == "test-id-1234"
        assert goal.description == "from dict"
        assert goal.max_steps == 50

    def test_goal_roundtrip(self) -> None:
        """Goal should survive serialization roundtrip."""
        original = Goal(
            description="roundtrip",
            success_criteria=["a", "b"],
            max_steps=25,
            invariants=["safety"],
            priority=3,
            metadata={"key": "val"},
        )
        data = original.model_dump()
        restored = Goal(**data)
        assert restored.description == original.description
        assert restored.max_steps == original.max_steps
        assert restored.success_criteria == original.success_criteria
        assert restored.invariants == original.invariants
        assert restored.priority == original.priority
        assert restored.metadata == original.metadata
