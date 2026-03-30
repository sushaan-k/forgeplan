"""Shared fixtures for agent-forge tests."""

from __future__ import annotations

from typing import Any

import pytest

from agent_forge.backtrack import BacktrackEngine
from agent_forge.executor import Executor, PlanStep
from agent_forge.goal import Goal
from agent_forge.models.base import BaseModel, ModelResponse
from agent_forge.monitor import Monitor
from agent_forge.state import StateManager
from agent_forge.tools.function import FunctionTool


class MockModel(BaseModel):
    """A mock LLM model for testing."""

    def __init__(
        self,
        model_name: str = "mock-model",
        responses: list[str] | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self._responses = list(responses or ["Mock response"])
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return a mock response."""
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        idx = min(self._call_index, len(self._responses) - 1)
        content = self._responses[idx]
        self._call_index += 1
        return ModelResponse(
            content=content,
            model=self.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


@pytest.fixture
def mock_model() -> MockModel:
    """Create a mock model that returns canned responses."""
    return MockModel()


@pytest.fixture
def state_manager() -> StateManager:
    """Create a fresh StateManager."""
    return StateManager()


@pytest.fixture
def monitor(state_manager: StateManager) -> Monitor:
    """Create a Monitor with no LLM (heuristic mode)."""
    return Monitor(state_manager=state_manager)


@pytest.fixture
def backtrack_engine(state_manager: StateManager) -> BacktrackEngine:
    """Create a BacktrackEngine."""
    return BacktrackEngine(state_manager=state_manager, max_depth=3)


@pytest.fixture
def executor(
    state_manager: StateManager,
    monitor: Monitor,
    backtrack_engine: BacktrackEngine,
) -> Executor:
    """Create an Executor with no LLM."""
    return Executor(
        state_manager=state_manager,
        monitor=monitor,
        backtrack_engine=backtrack_engine,
    )


@pytest.fixture
def sample_goal() -> Goal:
    """Create a sample goal for testing."""
    return Goal(
        description="Write a comprehensive market analysis report",
        success_criteria=[
            "Report contains data from at least 5 sources",
            "All claims are cited",
            "Report is saved to output directory",
        ],
        max_steps=20,
        invariants=[
            "Never fabricate data points",
            "Never overwrite existing files without confirmation",
        ],
    )


@pytest.fixture
def sample_steps() -> list[PlanStep]:
    """Create sample plan steps for testing."""
    return [
        PlanStep(
            id="step-1",
            description="Research market data",
            action="llm",
            postcondition="market data collected",
            expected_state_changes={"research_done": True},
        ),
        PlanStep(
            id="step-2",
            description="Analyze trends",
            action="llm",
            postcondition="trends identified",
            expected_state_changes={"analysis_done": True},
            depends_on=["step-1"],
        ),
        PlanStep(
            id="step-3",
            description="Write report",
            action="llm",
            postcondition="report written",
            expected_state_changes={"report_done": True},
            depends_on=["step-2"],
        ),
    ]


def sync_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def async_multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


@pytest.fixture
def sync_tool() -> FunctionTool:
    """Create a FunctionTool wrapping a sync function."""
    return FunctionTool(fn=sync_add, name="add")


@pytest.fixture
def async_tool() -> FunctionTool:
    """Create a FunctionTool wrapping an async function."""
    return FunctionTool(fn=async_multiply, name="multiply")
