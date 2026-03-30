"""Custom exceptions for agent-forge."""

from __future__ import annotations


class AgentForgeError(Exception):
    """Base exception for all agent-forge errors."""


class PlanningError(AgentForgeError):
    """Raised when the planner fails to produce a valid plan."""

    def __init__(self, message: str, goal_description: str = "") -> None:
        self.goal_description = goal_description
        super().__init__(
            f"Planning failed for goal '{goal_description}': {message}"
            if goal_description
            else f"Planning failed: {message}"
        )


class ExecutionError(AgentForgeError):
    """Raised when a plan step fails during execution."""

    def __init__(
        self, message: str, step_id: str = "", recoverable: bool = True
    ) -> None:
        self.step_id = step_id
        self.recoverable = recoverable
        super().__init__(
            f"Execution failed at step '{step_id}' "
            f"(recoverable={recoverable}): {message}"
        )


class BacktrackError(AgentForgeError):
    """Raised when backtracking fails or exceeds maximum depth."""

    def __init__(self, message: str, depth: int = 0, max_depth: int = 0) -> None:
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"Backtrack failed at depth {depth}/{max_depth}: {message}")


class InvariantViolation(AgentForgeError):
    """Raised when a global invariant is violated during execution."""

    def __init__(self, invariant: str, context: str = "") -> None:
        self.invariant = invariant
        self.context = context
        super().__init__(
            f"Invariant violated: '{invariant}'"
            + (f" (context: {context})" if context else "")
        )


class StateError(AgentForgeError):
    """Raised when state operations fail (checkpoint, rollback)."""


class ModelError(AgentForgeError):
    """Raised when an LLM API call fails."""

    def __init__(
        self, message: str, model: str = "", status_code: int | None = None
    ) -> None:
        self.model = model
        self.status_code = status_code
        detail = f"Model '{model}'" if model else "Model"
        if status_code:
            detail += f" (HTTP {status_code})"
        super().__init__(f"{detail}: {message}")


class ToolError(AgentForgeError):
    """Raised when a tool invocation fails."""

    def __init__(self, message: str, tool_name: str = "") -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' failed: {message}"
            if tool_name
            else f"Tool call failed: {message}"
        )
