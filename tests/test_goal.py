"""Tests for the Goal module."""

from __future__ import annotations

import pytest

from agent_forge.goal import Goal, GoalStatus


class TestGoal:
    """Tests for Goal creation and lifecycle."""

    def test_create_goal_with_defaults(self) -> None:
        """Goal should have sensible defaults."""
        goal = Goal(description="Test goal")
        assert goal.description == "Test goal"
        assert goal.max_steps == 100
        assert goal.status == GoalStatus.PENDING
        assert goal.success_criteria == []
        assert goal.invariants == []
        assert goal.priority == 0
        assert len(goal.id) == 12

    def test_create_goal_with_all_fields(self) -> None:
        """Goal should accept all configuration fields."""
        goal = Goal(
            description="Complex goal",
            success_criteria=["criterion 1", "criterion 2"],
            max_steps=50,
            invariants=["safety rule"],
            priority=5,
            metadata={"project": "test"},
        )
        assert goal.max_steps == 50
        assert len(goal.success_criteria) == 2
        assert len(goal.invariants) == 1
        assert goal.priority == 5
        assert goal.metadata["project"] == "test"

    def test_goal_status_transitions(self) -> None:
        """Goal should transition through lifecycle states."""
        goal = Goal(description="Lifecycle test")

        assert goal.status == GoalStatus.PENDING
        goal.mark_in_progress()
        assert goal.status == GoalStatus.IN_PROGRESS
        goal.mark_completed()
        assert goal.status == GoalStatus.COMPLETED

    def test_goal_failure_transition(self) -> None:
        """Goal should support failure transition."""
        goal = Goal(description="Failure test")
        goal.mark_in_progress()
        goal.mark_failed()
        assert goal.status == GoalStatus.FAILED

    def test_goal_cancellation(self) -> None:
        """Goal should support cancellation."""
        goal = Goal(description="Cancel test")
        goal.mark_cancelled()
        assert goal.status == GoalStatus.CANCELLED

    def test_goal_unique_ids(self) -> None:
        """Each goal should get a unique ID."""
        goals = [Goal(description=f"Goal {i}") for i in range(10)]
        ids = {g.id for g in goals}
        assert len(ids) == 10

    @pytest.mark.parametrize("max_steps", [1, 10, 500, 10000])
    def test_goal_max_steps_range(self, max_steps: int) -> None:
        """Goal should accept various max_steps values."""
        goal = Goal(description="Steps test", max_steps=max_steps)
        assert goal.max_steps == max_steps

    def test_goal_max_steps_validation(self) -> None:
        """Goal should reject invalid max_steps."""
        with pytest.raises(ValueError):
            Goal(description="Bad steps", max_steps=0)
        with pytest.raises(ValueError):
            Goal(description="Bad steps", max_steps=-1)
