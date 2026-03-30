"""Edge case tests for the MCTS plan selection module.

Covers single-node trees, deep trees, zero-visit nodes,
extreme exploration weights, heuristic rollout scoring, and
backpropagation corner cases.
"""

from __future__ import annotations

import math

import pytest

from agent_forge.executor import PlanStep
from agent_forge.goal import Goal
from agent_forge.planner import MCTSNode, MCTSSearch, PlanningError
from tests.conftest import MockModel


class TestMCTSNodeEdgeCases:
    """Edge cases for MCTSNode."""

    def test_average_reward_zero_visits(self) -> None:
        """average_reward with zero visits should return 0.0, not divide by zero."""
        node = MCTSNode(visits=0, total_reward=0.0)
        assert node.average_reward == 0.0

    def test_average_reward_single_visit(self) -> None:
        """average_reward with one visit should equal total_reward."""
        node = MCTSNode(visits=1, total_reward=0.75)
        assert node.average_reward == 0.75

    def test_average_reward_negative_total(self) -> None:
        """average_reward should handle negative total_reward."""
        node = MCTSNode(visits=2, total_reward=-0.4)
        assert node.average_reward == pytest.approx(-0.2)

    def test_ucb1_no_parent(self) -> None:
        """UCB1 with no parent should use self.visits for parent_visits."""
        node = MCTSNode(visits=5, total_reward=2.0)
        score = node.ucb1_score()
        exploitation = 2.0 / 5.0
        exploration = 1.41 * math.sqrt(math.log(5) / 5)
        assert score == pytest.approx(exploitation + exploration)

    def test_ucb1_zero_exploration_weight(self) -> None:
        """UCB1 with exploration_weight=0 should equal exploitation only."""
        parent = MCTSNode(visits=10)
        child = MCTSNode(visits=5, total_reward=3.0, parent=parent)
        score = child.ucb1_score(exploration_weight=0.0)
        assert score == pytest.approx(3.0 / 5.0)

    def test_ucb1_very_high_exploration_weight(self) -> None:
        """Very high exploration weight should dominate exploitation."""
        parent = MCTSNode(visits=100)
        child = MCTSNode(visits=1, total_reward=0.1, parent=parent)
        score = child.ucb1_score(exploration_weight=100.0)
        # Exploration term should be huge
        assert score > 100.0

    def test_node_unique_ids(self) -> None:
        """Each node should have a unique ID."""
        nodes = [MCTSNode() for _ in range(50)]
        ids = {n.id for n in nodes}
        assert len(ids) == 50

    def test_node_with_step(self) -> None:
        """Node should correctly store an associated PlanStep."""
        step = PlanStep(description="test step", postcondition="done")
        node = MCTSNode(step=step)
        assert node.step is step
        assert node.step.description == "test step"


class TestMCTSSearchEdgeCases:
    """Edge cases for MCTSSearch operations."""

    @pytest.fixture
    def goal(self) -> Goal:
        return Goal(description="Test goal", success_criteria=["done"], max_steps=10)

    @pytest.mark.asyncio
    async def test_search_empty_plans_raises(self, goal: Goal) -> None:
        """Should raise PlanningError with empty candidate list."""
        search = MCTSSearch(num_simulations=5)
        with pytest.raises(PlanningError, match="No candidate plans"):
            await search.search(goal, [])

    @pytest.mark.asyncio
    async def test_search_single_empty_plan(self, goal: Goal) -> None:
        """Single plan with no steps should be returned as-is."""
        search = MCTSSearch(num_simulations=5)
        empty_plan: list[PlanStep] = []
        result = await search.search(goal, [empty_plan])
        assert result == []

    @pytest.mark.asyncio
    async def test_search_single_step_plans(self, goal: Goal) -> None:
        """Plans with only one step each should be handled correctly."""
        search = MCTSSearch(num_simulations=10)
        plans = [
            [PlanStep(description="Plan A")],
            [PlanStep(description="Plan B")],
        ]
        result = await search.search(goal, plans)
        assert result in plans

    @pytest.mark.asyncio
    async def test_search_many_candidates(self, goal: Goal) -> None:
        """Should handle many candidate plans."""
        search = MCTSSearch(num_simulations=20)
        plans = [
            [PlanStep(description=f"Plan {i}", postcondition=f"p{i} done")]
            for i in range(20)
        ]
        result = await search.search(goal, plans)
        assert result in plans

    @pytest.mark.asyncio
    async def test_search_with_multi_step_plans(self, goal: Goal) -> None:
        """Should handle plans with many steps (deep untried_actions)."""
        search = MCTSSearch(num_simulations=15)
        long_plan = [
            PlanStep(description=f"Step {i}", postcondition=f"step {i} done")
            for i in range(10)
        ]
        short_plan = [PlanStep(description="Quick step")]
        result = await search.search(goal, [long_plan, short_plan])
        assert result in [long_plan, short_plan]

    @pytest.mark.asyncio
    async def test_search_zero_simulations(self, goal: Goal) -> None:
        """With zero simulations, should still return a plan (no crashes)."""
        search = MCTSSearch(num_simulations=0)
        plans = [
            [PlanStep(description="Plan A")],
            [PlanStep(description="Plan B")],
        ]
        result = await search.search(goal, plans)
        assert result in plans

    @pytest.mark.asyncio
    async def test_search_one_simulation(self, goal: Goal) -> None:
        """With one simulation, should return a valid plan."""
        search = MCTSSearch(num_simulations=1)
        plans = [
            [PlanStep(description="Plan A", postcondition="done")],
            [PlanStep(description="Plan B")],
        ]
        result = await search.search(goal, plans)
        assert result in plans


class TestMCTSHeuristicRollout:
    """Tests for the heuristic rollout function."""

    def test_heuristic_base_score(self) -> None:
        """Node with no step should get base score around 0.5."""
        node = MCTSNode(step=None)
        goal = Goal(description="test", max_steps=5)
        # Run many times due to randomness
        scores = [MCTSSearch._heuristic_rollout(node, goal) for _ in range(100)]
        avg = sum(scores) / len(scores)
        assert 0.3 < avg < 0.7

    def test_heuristic_with_postcondition(self) -> None:
        """Steps with postconditions should score higher on average."""
        goal = Goal(description="test", max_steps=5)
        with_post = MCTSNode(
            step=PlanStep(
                description="good step",
                postcondition="verified",
                expected_state_changes={"x": 1},
                preconditions=["ready"],
            )
        )
        without_post = MCTSNode(step=PlanStep(description="bare step"))

        scores_with = [
            MCTSSearch._heuristic_rollout(with_post, goal) for _ in range(200)
        ]
        scores_without = [
            MCTSSearch._heuristic_rollout(without_post, goal) for _ in range(200)
        ]
        assert sum(scores_with) / len(scores_with) > sum(scores_without) / len(
            scores_without
        )

    def test_heuristic_bounded_0_1(self) -> None:
        """Heuristic should always produce values in [0, 1]."""
        goal = Goal(description="test", max_steps=5)
        node = MCTSNode(
            step=PlanStep(
                description="x",
                postcondition="y",
                expected_state_changes={"a": 1},
                preconditions=["b"],
            )
        )
        for _ in range(500):
            score = MCTSSearch._heuristic_rollout(node, goal)
            assert 0.0 <= score <= 1.0


class TestMCTSBackpropagation:
    """Tests for backpropagation edge cases."""

    def test_backpropagate_single_node(self) -> None:
        """Backpropagating on a root node should update only that node."""
        root = MCTSNode()
        MCTSSearch._backpropagate(root, 0.5)
        assert root.visits == 1
        assert root.total_reward == 0.5

    def test_backpropagate_deep_chain(self) -> None:
        """Backpropagation should traverse a long chain to root."""
        nodes = [MCTSNode()]
        for _i in range(10):
            child = MCTSNode(parent=nodes[-1])
            nodes[-1].children.append(child)
            nodes.append(child)

        leaf = nodes[-1]
        MCTSSearch._backpropagate(leaf, 1.0)

        for node in nodes:
            assert node.visits == 1
            assert node.total_reward == 1.0

    def test_backpropagate_accumulates(self) -> None:
        """Multiple backpropagations should accumulate visits and reward."""
        root = MCTSNode()
        child = MCTSNode(parent=root)
        root.children.append(child)

        MCTSSearch._backpropagate(child, 0.3)
        MCTSSearch._backpropagate(child, 0.7)

        assert child.visits == 2
        assert child.total_reward == pytest.approx(1.0)
        assert root.visits == 2
        assert root.total_reward == pytest.approx(1.0)

    def test_backpropagate_zero_reward(self) -> None:
        """Backpropagating zero reward should still increment visits."""
        root = MCTSNode()
        child = MCTSNode(parent=root)
        root.children.append(child)

        MCTSSearch._backpropagate(child, 0.0)
        assert child.visits == 1
        assert child.total_reward == 0.0
        assert root.visits == 1


class TestMCTSSelectAndExpand:
    """Tests for select and expand operations."""

    def test_select_returns_leaf(self) -> None:
        """Select should return a leaf node."""
        search = MCTSSearch(num_simulations=5)
        root = MCTSNode(visits=8)  # parent needs visits > 0 for log
        child1 = MCTSNode(parent=root, visits=5, total_reward=2.5)
        child2 = MCTSNode(parent=root, visits=3, total_reward=1.0)
        root.children = [child1, child2]

        selected = search._select(root)
        # Should select one of the children (both are leaves)
        assert selected in [child1, child2]

    def test_select_with_zero_visit_parent_raises(self) -> None:
        """Select with zero-visit parent causes ValueError from math.log(0)."""
        search = MCTSSearch(num_simulations=5)
        root = MCTSNode(visits=0)
        child1 = MCTSNode(parent=root, visits=2, total_reward=1.0)
        child2 = MCTSNode(parent=root, visits=1, total_reward=0.5)
        root.children = [child1, child2]

        with pytest.raises(ValueError, match="math domain error"):
            search._select(root)

    def test_select_prefers_unvisited(self) -> None:
        """Select should prefer unvisited children (UCB1 = inf)."""
        search = MCTSSearch(num_simulations=5)
        root = MCTSNode(visits=10)
        visited = MCTSNode(parent=root, visits=5, total_reward=2.0)
        unvisited = MCTSNode(parent=root, visits=0, total_reward=0.0)
        root.children = [visited, unvisited]

        selected = search._select(root)
        assert selected is unvisited

    def test_expand_pops_action(self) -> None:
        """Expand should pop the first untried action."""
        search = MCTSSearch(num_simulations=5)
        action1 = PlanStep(description="action 1")
        action2 = PlanStep(description="action 2")
        node = MCTSNode(untried_actions=[action1, action2])

        child = search._expand(node)
        assert child.step is action1
        assert child.parent is node
        assert child in node.children
        assert len(node.untried_actions) == 1
        assert node.untried_actions[0] is action2


class TestMCTSWithModelRollouts:
    """Tests for MCTS with model-based rollouts."""

    @pytest.mark.asyncio
    async def test_model_rollout_invalid_score_falls_back(self) -> None:
        """When model returns non-numeric, should fall back to heuristic."""
        model = MockModel(responses=["not a number", "invalid"])
        search = MCTSSearch(
            model=model,
            rollout_model=model,
            num_simulations=3,
        )
        goal = Goal(description="test", max_steps=5)
        plans = [
            [PlanStep(description="A")],
            [PlanStep(description="B")],
        ]
        result = await search.search(goal, plans)
        assert result in plans

    @pytest.mark.asyncio
    async def test_model_rollout_score_clamping(self) -> None:
        """Scores outside [0, 1] should be clamped."""
        # Return values outside 0-1 range
        model = MockModel(responses=["2.5", "-0.5", "1.5", "0.3"])
        search = MCTSSearch(
            model=model,
            rollout_model=model,
            num_simulations=3,
        )
        goal = Goal(description="test", max_steps=5)
        plans = [
            [PlanStep(description="A")],
            [PlanStep(description="B")],
        ]
        # Should not crash even with out-of-range scores
        result = await search.search(goal, plans)
        assert result in plans
