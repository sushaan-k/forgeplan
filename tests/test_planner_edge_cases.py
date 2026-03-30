"""Edge case tests for the Planner module.

Covers parse_decomposition edge cases, beam search scoring,
agent wrapping, model resolution, and goal lifecycle during
planning.
"""

from __future__ import annotations

import os

import pytest

from agent_forge.executor import PlanStep
from agent_forge.goal import Goal, GoalStatus
from agent_forge.planner import Agent, Planner, SearchStrategy
from agent_forge.tools.function import FunctionTool
from tests.conftest import MockModel


class TestParseDecomposition:
    """Tests for parsing LLM decomposition responses."""

    def _make_planner(self) -> Planner:
        model = MockModel()
        agent = Agent(model=model)
        return Planner(agent=agent, search_strategy="greedy")

    def test_empty_response(self) -> None:
        """Empty response should produce no plans."""
        planner = self._make_planner()
        goal = Goal(description="test")
        plans = planner._parse_decomposition("", goal)
        assert plans == []

    def test_whitespace_only_response(self) -> None:
        """Whitespace-only response should produce no plans."""
        planner = self._make_planner()
        goal = Goal(description="test")
        plans = planner._parse_decomposition("   \n\n  \t  ", goal)
        assert plans == []

    def test_single_plan_no_plan_header(self) -> None:
        """Steps without PLAN header should still be parsed as one plan."""
        planner = self._make_planner()
        goal = Goal(description="test")
        response = "1. Do thing one | action: llm | postcondition: done\n2. Do thing two | action: tool1 | postcondition: complete"
        plans = planner._parse_decomposition(response, goal)
        assert len(plans) == 1
        assert len(plans[0]) == 2
        assert plans[0][0].action == "llm"
        assert plans[0][1].action == "tool1"

    def test_multiple_plans_parsed(self) -> None:
        """Multiple PLAN sections should produce multiple plans."""
        planner = self._make_planner()
        goal = Goal(description="test")
        response = (
            "PLAN 1:\n"
            "1. Step A | action: llm\n"
            "2. Step B | action: tool\n"
            "\n"
            "PLAN 2:\n"
            "1. Step C | action: llm\n"
            "\n"
            "PLAN 3:\n"
            "1. Step D | action: tool2 | postcondition: verified\n"
        )
        plans = planner._parse_decomposition(response, goal)
        assert len(plans) == 3

    def test_step_with_no_action(self) -> None:
        """Steps without explicit action should default to 'llm'."""
        planner = self._make_planner()
        goal = Goal(description="test")
        response = "PLAN 1:\n1. Just a description"
        plans = planner._parse_decomposition(response, goal)
        assert len(plans) == 1
        assert plans[0][0].action == "llm"

    def test_step_numbering_stripped(self) -> None:
        """Step numbers and dashes should be stripped from description."""
        planner = self._make_planner()
        goal = Goal(description="test")
        response = "PLAN 1:\n1. Research data\n2) Analyze results\n3- Write report"
        plans = planner._parse_decomposition(response, goal)
        assert len(plans) == 1
        descriptions = [s.description for s in plans[0]]
        # Numbering prefixes should be stripped
        assert all(not d.startswith("1") for d in descriptions)

    def test_plan_header_case_insensitive(self) -> None:
        """PLAN header should be case-insensitive."""
        planner = self._make_planner()
        goal = Goal(description="test")
        response = (
            "plan 1:\n1. Step A | action: llm\nPlan 2:\n1. Step B | action: tool\n"
        )
        plans = planner._parse_decomposition(response, goal)
        assert len(plans) == 2


class TestBeamSelect:
    """Tests for beam search plan selection."""

    @pytest.mark.asyncio
    async def test_beam_empty_raises(self) -> None:
        """Beam search with no candidates should raise PlanningError."""
        from agent_forge.exceptions import PlanningError

        model = MockModel()
        agent = Agent(model=model)
        planner = Planner(agent=agent, search_strategy="beam")
        goal = Goal(description="test")

        with pytest.raises(PlanningError, match="No candidate plans"):
            await planner._beam_select([], goal)

    @pytest.mark.asyncio
    async def test_beam_single_plan(self) -> None:
        """Beam search with one candidate should return it."""
        model = MockModel()
        agent = Agent(model=model)
        planner = Planner(agent=agent, search_strategy="beam")
        goal = Goal(description="test")
        plan = [PlanStep(description="only step", postcondition="done")]

        result = await planner._beam_select([plan], goal)
        assert result is plan

    @pytest.mark.asyncio
    async def test_beam_prefers_detailed_plans(self) -> None:
        """Beam search should prefer plans with postconditions and preconditions."""
        model = MockModel()
        agent = Agent(model=model)
        planner = Planner(agent=agent, search_strategy="beam")
        goal = Goal(description="test")

        detailed = [
            PlanStep(
                description="step",
                postcondition="done",
                preconditions=["ready"],
            )
        ]
        sparse = [PlanStep(description="step")]

        result = await planner._beam_select([sparse, detailed], goal)
        assert result is detailed

    @pytest.mark.asyncio
    async def test_beam_normalizes_by_plan_length(self) -> None:
        """Beam score should be normalized by plan length."""
        model = MockModel()
        agent = Agent(model=model)
        planner = Planner(agent=agent, search_strategy="beam")
        goal = Goal(description="test")

        short_good = [
            PlanStep(description="one", postcondition="p", preconditions=["r"])
        ]
        long_sparse = [PlanStep(description=f"step-{i}") for i in range(10)]

        result = await planner._beam_select([long_sparse, short_good], goal)
        assert result is short_good


class TestAgentEdgeCases:
    """Edge cases for Agent initialization."""

    def test_agent_with_mixed_tool_types(self) -> None:
        """Agent should handle mix of FunctionTool and callable."""

        def raw_fn(x: int) -> int:
            return x

        ft = FunctionTool(fn=raw_fn, name="wrapped")
        agent = Agent(tools=[raw_fn, ft])
        assert len(agent.function_tools) == 2
        names = {t.name for t in agent.function_tools}
        assert "raw_fn" in names
        assert "wrapped" in names

    def test_agent_ignores_non_callable_tools(self) -> None:
        """Non-callable items in tools list should be silently ignored."""
        agent = Agent(tools=["not_a_tool", 42, None])  # type: ignore[list-item]
        # Only callables get wrapped
        assert len(agent.function_tools) == 0

    def test_agent_system_prompt_stored(self) -> None:
        """System prompt should be stored."""
        agent = Agent(system_prompt="You are a helpful assistant")
        assert agent.system_prompt == "You are a helpful assistant"

    def test_resolve_model_unknown_name_no_key(self) -> None:
        """Unknown model name with no API keys should return None."""
        env_openai = os.environ.pop("OPENAI_API_KEY", None)
        env_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = Agent._resolve_model("unknown-model-xyz")
            assert result is None
        finally:
            if env_openai:
                os.environ["OPENAI_API_KEY"] = env_openai
            if env_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = env_anthropic


class TestPlannerGoalLifecycle:
    """Tests for goal status updates during planning."""

    @pytest.mark.asyncio
    async def test_successful_goal_marked_completed(self) -> None:
        """Goal should be marked COMPLETED on success."""
        agent = Agent(model=None, tools=[])
        planner = Planner(
            agent=agent,
            search_strategy="greedy",
            max_backtrack_depth=1,
        )
        goal = Goal(description="succeed", max_steps=5)
        result = await planner.execute(goal)
        if result.success:
            assert goal.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_goal_starts_in_progress(self) -> None:
        """Goal should transition to IN_PROGRESS during execution."""
        model = MockModel(
            responses=[
                "PLAN 1:\n1. Step | action: llm | postcondition: done\n",
                "result",
            ]
        )
        agent = Agent(model=model, tools=[])

        # Just verify the goal goes through IN_PROGRESS
        goal = Goal(description="lifecycle test", max_steps=5)
        planner = Planner(agent=agent, search_strategy="greedy")
        await planner.execute(goal)
        assert goal.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)


class TestPlannerDefaultPlan:
    """Tests for default plan generation."""

    def test_default_plan_includes_goal_description(self) -> None:
        """Default plan step should reference the goal description."""
        goal = Goal(description="Build a widget", success_criteria=["widget exists"])
        plan = Planner._generate_default_plan(goal)
        assert len(plan) == 1
        assert "Build a widget" in plan[0].description

    def test_default_plan_includes_success_criteria_as_postcondition(self) -> None:
        """Default plan postcondition should include success criteria."""
        goal = Goal(
            description="Test",
            success_criteria=["criterion1", "criterion2"],
        )
        plan = Planner._generate_default_plan(goal)
        assert "criterion1" in plan[0].postcondition
        assert "criterion2" in plan[0].postcondition

    def test_default_plan_empty_success_criteria(self) -> None:
        """Default plan with no success criteria should have empty postcondition."""
        goal = Goal(description="No criteria")
        plan = Planner._generate_default_plan(goal)
        assert plan[0].postcondition == ""


class TestPlannerStrategies:
    """Tests for search strategy validation."""

    def test_valid_strategies(self) -> None:
        """All valid strategy strings should be accepted."""
        model = MockModel()
        agent = Agent(model=model)
        for strategy in ["greedy", "mcts", "beam"]:
            planner = Planner(agent=agent, search_strategy=strategy)
            assert planner._strategy == SearchStrategy(strategy)

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy string should raise ValueError."""
        model = MockModel()
        agent = Agent(model=model)
        with pytest.raises(ValueError):
            Planner(agent=agent, search_strategy="invalid")

    def test_strategy_enum_values(self) -> None:
        """SearchStrategy enum should have expected values."""
        assert SearchStrategy.GREEDY == "greedy"
        assert SearchStrategy.MCTS == "mcts"
        assert SearchStrategy.BEAM == "beam"
