"""Goal definition and success criteria for agent-forge.

A Goal represents a high-level objective that the planning engine
decomposes into executable subtasks. Goals carry success criteria
(conditions that must be true when the goal is achieved) and
invariants (conditions that must remain true throughout execution).
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class GoalStatus(StrEnum):
    """Lifecycle status of a goal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Goal(BaseModel):
    """A high-level objective with success criteria and invariants.

    Args:
        description: Natural-language description of the goal.
        success_criteria: Conditions that must hold for the goal to be
            considered achieved. Evaluated after plan execution.
        max_steps: Upper bound on execution steps before forced termination.
        invariants: Conditions that must remain true at every step. A
            violation triggers backtracking and replanning.
        priority: Numeric priority (higher = more important). Used when
            multiple goals compete for resources.
        metadata: Arbitrary key-value pairs for user-defined context.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    success_criteria: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=100, ge=1)
    invariants: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0)
    status: GoalStatus = GoalStatus.PENDING
    metadata: dict[str, str] = Field(default_factory=dict)

    def mark_in_progress(self) -> None:
        """Transition the goal to in-progress status."""
        self.status = GoalStatus.IN_PROGRESS

    def mark_completed(self) -> None:
        """Transition the goal to completed status."""
        self.status = GoalStatus.COMPLETED

    def mark_failed(self) -> None:
        """Transition the goal to failed status."""
        self.status = GoalStatus.FAILED

    def mark_cancelled(self) -> None:
        """Transition the goal to cancelled status."""
        self.status = GoalStatus.CANCELLED
