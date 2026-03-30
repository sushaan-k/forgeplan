"""Tests for the MCTS plan selection module."""

from __future__ import annotations

import pytest

from agent_forge.executor import PlanStep
from agent_forge.goal import Goal
from agent_forge.planner import MCTSNode, MCTSSearch


class TestMCTSNode:
    """Tests for the MCTS tree node."""

    def test_initial_node(self) -> None:
        """New nodes should have zero visits and reward."""
        node = MCTSNode()
        assert node.visits == 0
        assert node.total_reward == 0.0
        assert node.average_reward == 0.0
        assert node.children == []

    def test_average_reward(self) -> None:
        """Average reward should compute correctly."""
        node = MCTSNode(visits=4, total_reward=2.0)
        assert node.average_reward == 0.5

    def test_ucb1_unvisited(self) -> None:
        """Unvisited nodes should have infinite UCB1 score."""
        node = MCTSNode(visits=0)
        assert node.ucb1_score() == float("inf")

    def test_ucb1_visited(self) -> None:
        """Visited nodes should produce finite UCB1 scores."""
        parent = MCTSNode(visits=10)
        child = MCTSNode(visits=3, total_reward=1.5, parent=parent)
        score = child.ucb1_score()
        assert 0 < score < float("inf")

    def test_ucb1_exploration_weight(self) -> None:
        """Higher exploration weight should increase UCB1 score."""
        parent = MCTSNode(visits=10)
        child = MCTSNode(visits=3, total_reward=1.5, parent=parent)
        low = child.ucb1_score(exploration_weight=0.5)
        high = child.ucb1_score(exploration_weight=2.0)
        assert high > low


class TestMCTSSearch:
    """Tests for MCTS plan search."""

    @pytest.fixture
    def search(self) -> MCTSSearch:
        """Create an MCTS search with heuristic rollouts."""
        return MCTSSearch(num_simulations=20, max_rollout_depth=5)

    @pytest.fixture
    def goal(self) -> Goal:
        """Create a test goal."""
        return Goal(
            description="Test goal",
            success_criteria=["success"],
            max_steps=10,
        )

    @pytest.fixture
    def candidate_plans(self) -> list[list[PlanStep]]:
        """Create candidate plans for testing."""
        plan_a = [
            PlanStep(
                description="Step A1",
                postcondition="A1 done",
                expected_state_changes={"a1": True},
            ),
            PlanStep(
                description="Step A2",
                postcondition="A2 done",
            ),
        ]
        plan_b = [
            PlanStep(description="Step B1"),
            PlanStep(description="Step B2"),
        ]
        return [plan_a, plan_b]

    @pytest.mark.asyncio
    async def test_search_single_plan(self, search: MCTSSearch, goal: Goal) -> None:
        """Should return the only plan when there is one candidate."""
        plan = [PlanStep(description="Only step")]
        result = await search.search(goal, [plan])
        assert result == plan

    @pytest.mark.asyncio
    async def test_search_multiple_plans(
        self, search: MCTSSearch, goal: Goal, candidate_plans: list[list[PlanStep]]
    ) -> None:
        """Should return one of the candidate plans."""
        result = await search.search(goal, candidate_plans)
        assert result in candidate_plans

    @pytest.mark.asyncio
    async def test_search_empty_raises(self, search: MCTSSearch, goal: Goal) -> None:
        """Should raise when no candidate plans provided."""
        from agent_forge.exceptions import PlanningError

        with pytest.raises(PlanningError):
            await search.search(goal, [])

    @pytest.mark.asyncio
    async def test_search_prefers_detailed_plans(self, goal: Goal) -> None:
        """Plans with postconditions should score higher on average."""
        search = MCTSSearch(num_simulations=100)
        detailed = [
            PlanStep(
                description="Detailed step",
                postcondition="verified",
                expected_state_changes={"done": True},
                preconditions=["ready"],
            )
        ]
        sparse = [PlanStep(description="Sparse step")]

        wins = {"detailed": 0, "sparse": 0}
        for _ in range(10):
            result = await search.search(goal, [detailed, sparse])
            if result == detailed:
                wins["detailed"] += 1
            else:
                wins["sparse"] += 1

        # Detailed plans should win more often than not
        assert wins["detailed"] >= wins["sparse"]

    def test_backpropagate(self) -> None:
        """Backpropagation should update all ancestors."""
        root = MCTSNode()
        child = MCTSNode(parent=root)
        grandchild = MCTSNode(parent=child)

        root.children.append(child)
        child.children.append(grandchild)

        MCTSSearch._backpropagate(grandchild, 0.8)
        assert grandchild.visits == 1
        assert grandchild.total_reward == 0.8
        assert child.visits == 1
        assert child.total_reward == 0.8
        assert root.visits == 1
        assert root.total_reward == 0.8
