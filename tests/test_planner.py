"""Tests for the Planner module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_forge.executor import ExecutionResult
from agent_forge.goal import Goal, GoalStatus
from agent_forge.planner import Agent, Planner, SearchStrategy
from agent_forge.tools.mcp import MCPTool
from tests.conftest import MockModel


class TestAgent:
    """Tests for the Agent wrapper."""

    def test_create_agent_no_model(self) -> None:
        """Agent should handle None model gracefully."""
        agent = Agent(model=None, tools=[])
        assert agent.model is None
        assert agent.function_tools == []

    def test_create_agent_with_callable_tools(self) -> None:
        """Agent should wrap raw callables as FunctionTools."""

        def my_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[my_tool])
        assert len(agent.function_tools) == 1
        assert agent.function_tools[0].name == "my_tool"

    def test_create_agent_with_model_instance(self) -> None:
        """Agent should accept a BaseModel instance directly."""
        model = MockModel()
        agent = Agent(model=model)
        assert agent.model is model

    def test_create_agent_with_string_model_no_key(self) -> None:
        """Agent with string model and no API key returns None model."""
        import os

        env_backup_openai = os.environ.pop("OPENAI_API_KEY", None)
        env_backup_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            agent = Agent(model="gpt-4o")
            assert agent.model is None
        finally:
            if env_backup_openai:
                os.environ["OPENAI_API_KEY"] = env_backup_openai
            if env_backup_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = env_backup_anthropic

    def test_create_agent_preserves_mcp_tools(self) -> None:
        """Agent should keep MCPTool instances instead of dropping them."""
        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
            description="Search the web",
        )
        agent = Agent(tools=[tool])
        assert agent.tools[0] is tool
        assert agent.function_tools[0] is tool


class TestPlanner:
    """Tests for the Planner orchestration."""

    @pytest.fixture
    def mock_agent(self) -> Agent:
        """Create an agent with a mock model."""
        model = MockModel(
            responses=[
                "PLAN 1:\n1. Research data | action: llm | postcondition: data collected\n2. Write report | action: llm | postcondition: report done\n"
            ]
        )
        return Agent(model=model, tools=[])

    @pytest.fixture
    def planner(self, mock_agent: Agent) -> Planner:
        """Create a Planner with mock agent."""
        return Planner(
            agent=mock_agent,
            search_strategy="greedy",
            max_backtrack_depth=2,
            checkpoint_interval=2,
        )

    @pytest.mark.asyncio
    async def test_execute_simple_goal(self, planner: Planner) -> None:
        """Planner should execute a simple goal."""
        goal = Goal(
            description="Simple task",
            success_criteria=["done"],
            max_steps=10,
        )
        result = await planner.execute(goal)
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_system_prompt_and_tools_reach_model(self) -> None:
        """Planner should pass system prompt and tool schemas to the model."""

        model = MockModel(
            responses=[
                "PLAN 1:\n1. Search | action: fetch | postcondition: found\n",
                "Search complete",
            ]
        )
        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="fetch",
            description="Fetch data",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        tool.execute = AsyncMock(return_value="fetched")  # type: ignore[assignment]
        agent = Agent(
            model=model,
            tools=[tool],
            system_prompt="You are a careful planner.",
        )
        planner = Planner(agent=agent, search_strategy="greedy")

        goal = Goal(description="Search data", max_steps=5)
        result = await planner.execute(goal)

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert model.calls[0]["messages"][0]["role"] == "system"
        assert model.calls[0]["messages"][0]["content"] == "You are a careful planner."
        assert model.calls[0]["tools"][0]["function"]["name"] == "fetch"
        assert "fetch" in planner._executor._tools

    @pytest.mark.asyncio
    async def test_goal_status_updated(self, planner: Planner) -> None:
        """Goal status should be updated during execution."""
        goal = Goal(description="Status test", max_steps=5)
        await planner.execute(goal)
        assert goal.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)

    def test_planner_strategies(self) -> None:
        """Planner should accept different search strategies."""
        model = MockModel()
        agent = Agent(model=model)

        for strategy in ["greedy", "mcts", "beam"]:
            planner = Planner(agent=agent, search_strategy=strategy)
            assert planner._strategy == SearchStrategy(strategy)

    def test_parse_decomposition(self) -> None:
        """Should parse LLM plan decomposition responses."""
        model = MockModel()
        agent = Agent(model=model)
        planner = Planner(agent=agent)

        response = (
            "PLAN 1:\n"
            "1. First step | action: tool1 | postcondition: step 1 done\n"
            "2. Second step | action: llm | postcondition: step 2 done\n"
            "\n"
            "PLAN 2:\n"
            "1. Alt step | action: tool2 | postcondition: alt done\n"
        )

        goal = Goal(description="test")
        plans = planner._parse_decomposition(response, goal)
        assert len(plans) >= 1

    def test_generate_default_plan(self) -> None:
        """Default plan should have a single step."""
        goal = Goal(
            description="Default test",
            success_criteria=["criterion1"],
        )
        plan = Planner._generate_default_plan(goal)
        assert len(plan) == 1
        assert "Default test" in plan[0].description

    @pytest.mark.asyncio
    async def test_execute_no_model(self) -> None:
        """Planner with no LLM should use default plan."""
        agent = Agent(model=None, tools=[])
        planner = Planner(
            agent=agent,
            search_strategy="greedy",
            max_backtrack_depth=1,
        )
        goal = Goal(description="No model test", max_steps=5)
        result = await planner.execute(goal)
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_max_steps_is_enforced(self) -> None:
        """Planner should stop once Goal.max_steps is reached."""
        model = MockModel(
            responses=[
                "PLAN 1:\n1. Step one | action: llm\n2. Step two | action: llm\n",
                "step one result",
                "step two result",
            ]
        )
        agent = Agent(model=model, tools=[])
        planner = Planner(agent=agent, search_strategy="greedy")
        goal = Goal(description="Limited goal", max_steps=1)

        result = await planner.execute(goal)

        assert result.success is False
        assert result.steps_completed == 1
        assert "max_steps" in result.error
        assert result.recoverable is False
        assert goal.status == GoalStatus.FAILED
