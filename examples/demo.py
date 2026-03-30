#!/usr/bin/env python3
"""Offline demo for agent-forge."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_forge import Agent, Goal, Planner
from agent_forge.models.base import BaseModel, ModelResponse


def gather_metrics() -> dict[str, str]:
    return {"research_summary": "Collected benchmark signals for the launch brief."}


def draft_brief() -> dict[str, str]:
    return {"brief": "Prepared a portfolio brief for senior engineering reviewers."}


class DemoModel(BaseModel):
    def __init__(self) -> None:
        super().__init__(model_name="demo-model")

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        prompt = messages[-1]["content"]
        if "Provide 1-3 alternative plans" in prompt:
            return ModelResponse(
                content=(
                    "PLAN 1:\n"
                    "1. Gather benchmark signals | action: gather_metrics | "
                    "postcondition: benchmark summary captured\n"
                    "2. Draft a final portfolio brief | action: draft_brief | "
                    "postcondition: brief completed\n"
                ),
                model=self.model_name,
            )
        return ModelResponse(content="step completed", model=self.model_name)


async def main() -> None:
    goal = Goal(
        description="Assemble a concise launch brief for an open-source AI system.",
        success_criteria=["Benchmark summary captured", "Brief completed"],
        max_steps=5,
    )
    agent = Agent(
        model=DemoModel(),
        tools=[gather_metrics, draft_brief],
        system_prompt="You are a planning engine that prefers concrete actions.",
    )
    planner = Planner(agent=agent, search_strategy="beam", num_simulations=8)
    result = await planner.execute(goal)

    print("agent-forge demo")
    print(f"success: {result.success}")
    print(f"steps completed: {result.steps_completed}/{result.steps_total}")
    print(f"final state keys: {sorted(result.final_state.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
