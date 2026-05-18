"""agent-forge: A long-horizon planning engine for LLM agents.

Provides HTN decomposition, MCTS plan selection, and backtracking
with causal dependency reasoning to solve the "early myopic commitment"
problem in LLM agent planning.

Example:
    >>> from agent_forge import Planner, Agent, Goal
    >>> goal = Goal(
    ...     description="Write a market analysis report",
    ...     success_criteria=["Report saved to output/"],
    ...     max_steps=20,
    ... )
    >>> agent = Agent(model="gpt-4o", tools=[])
    >>> planner = Planner(agent=agent, search_strategy="mcts")
    >>> result = await planner.execute(goal)
"""

from agent_forge.backtrack import BacktrackEngine, BacktrackEvent
from agent_forge.exceptions import (
    AgentForgeError,
    BacktrackError,
    ExecutionError,
    InvariantViolation,
    ModelError,
    PlanningError,
    StateError,
    ToolError,
)
from agent_forge.executor import (
    ExecutionEvent,
    ExecutionResult,
    Executor,
    PlanStep,
    StepResult,
)
from agent_forge.goal import Goal, GoalStatus
from agent_forge.monitor import CheckResult, Monitor, MonitorVerdict
from agent_forge.planner import (
    STRATEGY_REGISTRY,
    Agent,
    MCTSSearch,
    Planner,
    SearchStrategy,
    register_strategy,
)
from agent_forge.state import Checkpoint, StateManager
from agent_forge.tools.function import FunctionTool
from agent_forge.tools.mcp import MCPTool

__version__ = "0.1.0"

__all__ = [
    "STRATEGY_REGISTRY",
    "Agent",
    "AgentForgeError",
    "BacktrackEngine",
    "BacktrackError",
    "BacktrackEvent",
    "CheckResult",
    "Checkpoint",
    "ExecutionError",
    "ExecutionEvent",
    "ExecutionResult",
    "Executor",
    "FunctionTool",
    "Goal",
    "GoalStatus",
    "InvariantViolation",
    "MCPTool",
    "MCTSSearch",
    "ModelError",
    "Monitor",
    "MonitorVerdict",
    "PlanStep",
    "Planner",
    "PlanningError",
    "SearchStrategy",
    "StateError",
    "StateManager",
    "StepResult",
    "ToolError",
    "register_strategy",
]
