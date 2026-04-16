"""HTN planner with MCTS plan selection.

The Planner decomposes high-level Goals into executable PlanStep trees
using Hierarchical Task Network (HTN) decomposition, then selects
among candidate plans using Monte Carlo Tree Search (MCTS).
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from agent_forge.backtrack import BacktrackEngine
from agent_forge.exceptions import PlanningError
from agent_forge.executor import ExecutionResult, Executor, PlanStep
from agent_forge.monitor import Monitor
from agent_forge.state import StateManager
from agent_forge.tools.function import FunctionTool

if TYPE_CHECKING:
    from agent_forge.goal import Goal
    from agent_forge.models.base import BaseModel as LLMBaseModel

logger = logging.getLogger(__name__)


class SearchStrategy(StrEnum):
    """Plan search strategies."""

    GREEDY = "greedy"
    MCTS = "mcts"
    BEAM = "beam"


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        id: Unique node identifier.
        step: The plan step this node represents.
        parent: Parent node (None for root).
        children: Child nodes (alternative next steps).
        visits: Number of times this node has been visited.
        total_reward: Cumulative reward from simulations through this node.
        untried_actions: Actions that haven't been expanded yet.
    """

    id: str = dc_field(default_factory=lambda: uuid.uuid4().hex[:8])
    step: PlanStep | None = None
    parent: MCTSNode | None = None
    children: list[MCTSNode] = dc_field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[PlanStep] = dc_field(default_factory=list)

    @property
    def average_reward(self) -> float:
        """Average reward across all visits."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1_score(self, exploration_weight: float = 1.41) -> float:
        """Compute the UCB1 score for node selection.

        Balances exploitation (high average reward) with exploration
        (under-visited nodes).

        Args:
            exploration_weight: Controls exploration vs exploitation.

        Returns:
            UCB1 score (higher = more promising to explore).
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.average_reward
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration


class MCTSSearch:
    """Monte Carlo Tree Search for plan selection.

    Builds a search tree of possible plan sequences and uses
    lightweight rollouts (via a fast LLM) to estimate which
    plan path is most likely to succeed.

    Args:
        model: LLM for generating candidate actions and rollouts.
        rollout_model: Cheaper/faster LLM for simulation rollouts.
        num_simulations: Number of MCTS iterations to run.
        max_rollout_depth: Maximum depth for rollout simulations.
        exploration_weight: UCB1 exploration constant.
    """

    def __init__(
        self,
        model: LLMBaseModel | None = None,
        rollout_model: LLMBaseModel | None = None,
        num_simulations: int = 50,
        max_rollout_depth: int = 10,
        exploration_weight: float = 1.41,
    ) -> None:
        self._model = model
        self._rollout_model = rollout_model or model
        self._num_simulations = num_simulations
        self._max_rollout_depth = max_rollout_depth
        self._exploration_weight = exploration_weight
        logger.debug(
            "MCTSSearch initialized (simulations=%d, max_depth=%d)",
            num_simulations,
            max_rollout_depth,
        )

    async def search(
        self,
        goal: Goal,
        candidate_plans: list[list[PlanStep]],
    ) -> list[PlanStep]:
        """Select the best plan using MCTS.

        Evaluates candidate plans through simulated rollouts and
        returns the plan with the highest expected reward.

        Args:
            goal: The goal being planned for.
            candidate_plans: List of candidate plan step sequences.

        Returns:
            The best plan (list of PlanStep).
        """
        if not candidate_plans:
            raise PlanningError(
                message="No candidate plans to evaluate",
                goal_description=goal.description,
            )

        if len(candidate_plans) == 1:
            return candidate_plans[0]

        root = MCTSNode()
        for plan in candidate_plans:
            child = MCTSNode(
                step=plan[0] if plan else None,
                parent=root,
                untried_actions=plan[1:] if len(plan) > 1 else [],
            )
            root.children.append(child)

        for _ in range(self._num_simulations):
            node = self._select(root)
            if node.untried_actions:
                node = self._expand(node)
            reward = await self._simulate(node, goal)
            self._backpropagate(node, reward)

        best_child = max(root.children, key=lambda c: c.average_reward)
        best_idx = root.children.index(best_child)
        logger.info(
            "MCTS selected plan %d/%d (avg_reward=%.3f, visits=%d)",
            best_idx + 1,
            len(candidate_plans),
            best_child.average_reward,
            best_child.visits,
        )
        return candidate_plans[best_idx]

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1.

        Args:
            node: Starting node.

        Returns:
            Selected leaf node.
        """
        while node.children and not node.untried_actions:
            node = max(
                node.children,
                key=lambda c: c.ucb1_score(self._exploration_weight),
            )
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding one untried action.

        Args:
            node: Node to expand.

        Returns:
            The newly created child node.
        """
        action = node.untried_actions.pop(0)
        child = MCTSNode(step=action, parent=node)
        node.children.append(child)
        return child

    async def _simulate(self, node: MCTSNode, goal: Goal) -> float:
        """Simulate a rollout from the given node.

        Uses heuristics and optionally the rollout LLM to estimate
        the probability of success from this point.

        Args:
            node: Starting node for the simulation.
            goal: The goal to evaluate against.

        Returns:
            Reward value between 0.0 and 1.0.
        """
        if self._rollout_model is None:
            return self._heuristic_rollout(node, goal)

        step_descriptions: list[str] = []
        current: MCTSNode | None = node
        depth = 0
        while current and current.step and depth < self._max_rollout_depth:
            step_descriptions.append(current.step.description)
            current = current.parent
            depth += 1

        step_descriptions.reverse()

        prompt = (
            f"You are evaluating whether a plan will succeed.\n\n"
            f"Goal: {goal.description}\n"
            f"Success criteria: {goal.success_criteria}\n"
            f"Plan steps so far:\n"
            + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(step_descriptions))
            + "\n\nRate the probability this plan will succeed from 0.0 to 1.0. "
            "Answer with ONLY a number."
        )

        try:
            response = await self._rollout_model.generate_text(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, Exception):
            return self._heuristic_rollout(node, goal)

    @staticmethod
    def _heuristic_rollout(node: MCTSNode, goal: Goal) -> float:
        """Fast heuristic rollout when no LLM is available.

        Estimates plan quality based on structural properties:
        number of steps, presence of postconditions, etc.

        Args:
            node: Node to evaluate.
            goal: The goal.

        Returns:
            Heuristic reward between 0.0 and 1.0.
        """
        score = 0.5

        if node.step:
            if node.step.postcondition:
                score += 0.1
            if node.step.preconditions:
                score += 0.05
            if node.step.expected_state_changes:
                score += 0.1
            if node.step.description:
                score += 0.05

        noise = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, score + noise))

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        """Propagate the simulation reward up the tree.

        Args:
            node: Leaf node where simulation ended.
            reward: The simulation reward.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent


class Planner:
    """High-level planning engine combining HTN decomposition and MCTS.

    The Planner is the main entry point for agent-forge. It:
    1. Decomposes a Goal into candidate plans (HTN).
    2. Selects the best plan (MCTS or greedy).
    3. Executes the plan with monitoring and backtracking.
    4. Replans if backtracking occurs.

    Args:
        agent: An Agent instance wrapping an LLM and tools.
        search_strategy: Plan selection strategy.
        max_backtrack_depth: Maximum consecutive backtracks.
        checkpoint_interval: Steps between checkpoints.
        rollout_model: Cheaper LLM for MCTS rollouts.
        num_simulations: MCTS simulation count.
    """

    def __init__(
        self,
        agent: Agent,
        search_strategy: str | SearchStrategy = SearchStrategy.MCTS,
        max_backtrack_depth: int = 5,
        checkpoint_interval: int = 3,
        rollout_model: LLMBaseModel | None = None,
        num_simulations: int = 50,
    ) -> None:
        self._agent = agent
        self._strategy = SearchStrategy(search_strategy)
        self._max_backtrack_depth = max_backtrack_depth
        self._checkpoint_interval = checkpoint_interval

        self._state_manager = StateManager()
        self._monitor = Monitor(
            state_manager=self._state_manager,
            model=agent.model,
            system_prompt=agent.system_prompt,
        )
        self._backtrack_engine = BacktrackEngine(
            state_manager=self._state_manager,
            max_depth=max_backtrack_depth,
        )
        self._mcts = MCTSSearch(
            model=agent.model,
            rollout_model=rollout_model,
            num_simulations=num_simulations,
        )
        self._executor = Executor(
            state_manager=self._state_manager,
            monitor=self._monitor,
            backtrack_engine=self._backtrack_engine,
            model=agent.model,
            tools={t.name: t for t in agent.tools if hasattr(t, "name")},
            checkpoint_interval=checkpoint_interval,
            system_prompt=agent.system_prompt,
        )
        logger.info(
            "Planner initialized (strategy=%s, max_backtrack=%d)",
            self._strategy.value,
            max_backtrack_depth,
        )

    async def execute(self, goal: Goal) -> ExecutionResult:
        """Decompose, plan, and execute a goal.

        This is the main entry point. It generates a plan from the
        goal, selects the best candidate, executes it, and handles
        replanning on backtrack.

        Args:
            goal: The high-level goal to achieve.

        Returns:
            ExecutionResult with success status and execution details.

        Raises:
            PlanningError: If no valid plan can be generated.
        """
        degraded = self._agent.model is None
        if degraded:
            logger.warning(
                "No LLM model available. The planner is running in degraded "
                "mode with a trivial default plan. Set OPENAI_API_KEY or "
                "ANTHROPIC_API_KEY for full planning capability."
            )

        goal.mark_in_progress()
        logger.info("Planning goal: %s", goal.description[:100])

        plan_attempts = 0
        max_replan_attempts = self._max_backtrack_depth + 1
        all_plan_traces: list[PlanStep] = []

        while plan_attempts < max_replan_attempts:
            plan_attempts += 1
            logger.info("Plan attempt %d/%d", plan_attempts, max_replan_attempts)

            candidates = await self._decompose_goal(goal)
            if not candidates:
                raise PlanningError(
                    message="Failed to generate any candidate plans",
                    goal_description=goal.description,
                )

            if self._strategy == SearchStrategy.MCTS and len(candidates) > 1:
                selected_plan = await self._mcts.search(goal, candidates)
            elif self._strategy == SearchStrategy.BEAM and len(candidates) > 1:
                selected_plan = await self._beam_select(candidates, goal)
            else:
                selected_plan = candidates[0]

            all_plan_traces.extend(selected_plan)

            result = await self._executor.execute_plan(
                steps=selected_plan,
                invariants=goal.invariants,
                max_steps=goal.max_steps,
            )
            result.degraded = degraded
            result.plan_trace = all_plan_traces
            result.backtrack_log = [
                e.model_dump() for e in self._backtrack_engine.events
            ]

            if result.success:
                if degraded and goal.success_criteria:
                    goal.mark_failed()
                    result.success = False
                    result.error = "Degraded mode cannot verify goal success criteria"
                    return result
                goal.mark_completed()
                logger.info(
                    "Goal completed successfully in %d steps", result.steps_completed
                )
                return result

            if not result.recoverable:
                goal.mark_failed()
                logger.warning("Goal failed: execution was not recoverable")
                return result

            if not self._backtrack_engine.can_backtrack():
                goal.mark_failed()
                logger.warning("Goal failed: cannot backtrack further")
                return result

            logger.info("Replanning after backtrack (attempt %d)", plan_attempts)

        goal.mark_failed()
        return ExecutionResult(
            success=False,
            degraded=degraded,
            steps_completed=0,
            steps_total=0,
            final_state=self._state_manager.current_state,
            error=f"Exceeded max replan attempts ({max_replan_attempts})",
            plan_trace=all_plan_traces,
            backtrack_log=[e.model_dump() for e in self._backtrack_engine.events],
            recoverable=False,
        )

    async def _decompose_goal(self, goal: Goal) -> list[list[PlanStep]]:
        """Decompose a goal into candidate plans using HTN.

        Uses the LLM to propose task decompositions and generates
        multiple candidate plans for MCTS evaluation.

        Args:
            goal: The goal to decompose.

        Returns:
            List of candidate plans (each plan is a list of PlanStep).
        """
        if self._agent.model is None:
            return [self._generate_default_plan(goal)]

        prompt = self._build_decomposition_prompt(goal)
        response = await self._agent.model.generate_text(
            prompt,
            system=self._agent.system_prompt or None,
            tools=self._agent.tool_schemas,
        )
        plans = self._parse_decomposition(response, goal)

        if not plans:
            plans = [self._generate_default_plan(goal)]

        logger.info("Generated %d candidate plans", len(plans))
        return plans

    def _build_decomposition_prompt(self, goal: Goal) -> str:
        """Build the HTN decomposition prompt.

        Args:
            goal: The goal to decompose.

        Returns:
            Prompt string for the LLM.
        """
        state_summary = self._state_manager.current_state
        tools_desc = ", ".join(t.name for t in self._agent.function_tools)

        return (
            f"You are a task planning system. Decompose this goal into "
            f"a sequence of concrete, executable steps.\n\n"
            f"Goal: {goal.description}\n"
            f"Success criteria: {goal.success_criteria}\n"
            f"Invariants (must never be violated): {goal.invariants}\n"
            f"Max steps allowed: {goal.max_steps}\n"
            f"Available tools: {tools_desc or 'none'}\n"
            f"Current state: {state_summary}\n\n"
            f"Provide 1-3 alternative plans. For each plan, list steps as:\n"
            f"PLAN 1:\n"
            f"1. [step description] | action: [tool_name or 'llm'] | "
            f"postcondition: [what should be true after]\n"
            f"2. ...\n\n"
            f"PLAN 2:\n"
            f"1. ...\n"
        )

    def _parse_decomposition(self, response: str, goal: Goal) -> list[list[PlanStep]]:
        """Parse the LLM's decomposition response into PlanStep lists.

        Args:
            response: Raw LLM response text.
            goal: The goal (for context).

        Returns:
            List of candidate plans parsed from the response.
        """
        plans: list[list[PlanStep]] = []
        current_plan: list[PlanStep] = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("PLAN"):
                if current_plan:
                    plans.append(current_plan)
                current_plan = []
                continue

            parts = line.split("|")
            desc = parts[0].strip().lstrip("0123456789.-) ")
            action = "llm"
            postcondition = ""

            for part in parts[1:]:
                part = part.strip()
                if part.lower().startswith("action:"):
                    action = part.split(":", 1)[1].strip()
                elif part.lower().startswith("postcondition:"):
                    postcondition = part.split(":", 1)[1].strip()

            if desc:
                step = PlanStep(
                    description=desc,
                    action=action,
                    postcondition=postcondition,
                )
                current_plan.append(step)

        if current_plan:
            plans.append(current_plan)

        return plans

    @staticmethod
    def _generate_default_plan(goal: Goal) -> list[PlanStep]:
        """Generate a simple single-step fallback plan.

        Args:
            goal: The goal to plan for.

        Returns:
            A single-step plan.
        """
        return [
            PlanStep(
                description=f"Execute goal: {goal.description}",
                action="llm",
                postcondition="; ".join(goal.success_criteria),
            )
        ]

    async def _beam_select(
        self,
        candidates: list[list[PlanStep]],
        goal: Goal,
    ) -> list[PlanStep]:
        """Select the best plan using beam search scoring.

        Simpler than MCTS -- scores each plan independently and
        picks the highest scorer.

        Args:
            candidates: Candidate plans.
            goal: The goal.

        Returns:
            The best-scoring plan.
        """
        if not candidates:
            raise PlanningError(
                message="No candidate plans for beam search",
                goal_description=goal.description,
            )

        scores: list[float] = []
        for plan in candidates:
            score = 0.0
            for step in plan:
                if step.postcondition:
                    score += 0.2
                if step.preconditions:
                    score += 0.1
                if step.description:
                    score += 0.1
            score /= max(len(plan), 1)
            scores.append(score)

        best_idx = int(np.argmax(scores))
        logger.info(
            "Beam search selected plan %d/%d (score=%.3f)",
            best_idx + 1,
            len(candidates),
            scores[best_idx],
        )
        return candidates[best_idx]


class Agent:
    """Wraps an LLM and tools into an agent that the Planner can drive.

    This is the user-facing agent configuration object.

    Args:
        model: Model identifier string or BaseModel instance.
        tools: List of tool objects (FunctionTool, MCPTool, or callables).
        system_prompt: Optional system prompt for the LLM.
    """

    def __init__(
        self,
        model: str | LLMBaseModel | None = None,
        tools: list[Any] | None = None,
        system_prompt: str = "",
    ) -> None:
        if isinstance(model, str):
            self._model = self._resolve_model(model)
        else:
            self._model = model
        self._tools = tools or []
        self.system_prompt = system_prompt
        self._function_tools = self._wrap_tools(self._tools)

    @property
    def model(self) -> LLMBaseModel | None:
        """Return the underlying LLM model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return all configured tools, including MCP tools."""
        return list(self._function_tools)

    @property
    def function_tools(self) -> list[Any]:
        """Backward-compatible alias for all wrapped tools."""
        return list(self._function_tools)

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all known tools."""
        schemas: list[dict[str, Any]] = []
        for tool in self._function_tools:
            to_schema = getattr(tool, "to_schema", None)
            if callable(to_schema):
                schemas.append(to_schema())
        return schemas

    @staticmethod
    def _resolve_model(model_name: str) -> LLMBaseModel | None:
        """Resolve a model name string to a model backend.

        Args:
            model_name: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-6").

        Returns:
            Appropriate model backend, or None if no API key available.
        """
        import os
        import re

        name_lower = model_name.lower()

        # Anthropic models: names containing "claude" or "anthropic"
        _anthropic_pattern = re.compile(r"(?:claude|anthropic)")
        # OpenAI models: names starting with known prefixes like "gpt",
        # "o1", "o3", "o4", "chatgpt", or "openai"
        _openai_pattern = re.compile(
            r"^(?:gpt|o[0-9]|chatgpt|openai|davinci|babbage|text-)"
        )

        if _anthropic_pattern.search(name_lower):
            if os.environ.get("ANTHROPIC_API_KEY"):
                from agent_forge.models.anthropic import AnthropicModel

                return AnthropicModel(model_name=model_name)
        elif _openai_pattern.search(name_lower) and os.environ.get("OPENAI_API_KEY"):
            from agent_forge.models.openai import OpenAIModel

            return OpenAIModel(model_name=model_name)

        logger.warning(
            "No API key found for model '%s'. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
            model_name,
        )
        return None

    @staticmethod
    def _wrap_tools(tools: list[Any]) -> list[Any]:
        """Wrap raw callables as FunctionTool instances.

        Args:
            tools: List of tools (FunctionTool, callable, or other).

        Returns:
            List of FunctionTool instances.
        """
        wrapped: list[Any] = []
        for tool in tools:
            if isinstance(tool, FunctionTool) or (
                hasattr(tool, "execute") and hasattr(tool, "name")
            ):
                wrapped.append(tool)
            elif callable(tool):
                wrapped.append(FunctionTool(fn=tool))
        return wrapped
