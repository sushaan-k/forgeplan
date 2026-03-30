"""Research agent example -- plan, execute, and write a market report.

Demonstrates the full agent-forge planning loop:
1. Define a Goal with success criteria and invariants.
2. Register tools (here, simple Python functions).
3. Create a Planner and execute the goal.
4. Inspect the execution result and backtrack log.

Requirements:
    pip install agent-forge
    export ANTHROPIC_API_KEY="sk-..." or OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import asyncio

from agent_forge import Agent, Goal, Planner

# --- Tool definitions ---
# These would normally call real APIs. The planner wraps them automatically.


def search_sources(query: str) -> str:
    """Search for market research sources on a given topic."""
    # In production, this would call a search API
    return (
        f"Found 5 sources for '{query}': "
        f"(1) Bloomberg report Q4, "
        f"(2) McKinsey industry brief, "
        f"(3) S&P Capital IQ data, "
        f"(4) Federal Reserve economic data, "
        f"(5) Company 10-K filings"
    )


def fetch_data(source: str) -> str:
    """Fetch data from a specific source."""
    return f"Data retrieved from {source}: [sample dataset with 50 records]"


def write_section(title: str, content: str) -> str:
    """Write a section of the report."""
    return f"## {title}\n\n{content}\n"


def save_report(filename: str, content: str) -> str:
    """Save the final report to disk."""
    # In production, this would write to the filesystem
    return f"Report saved to output/{filename} ({len(content)} chars)"


async def main() -> None:
    """Run the research agent."""
    # 1. Define the goal with success criteria and safety invariants
    goal = Goal(
        description="Research and write a comprehensive market analysis report on the AI infrastructure market",
        success_criteria=[
            "Report contains data from at least 5 sources",
            "All claims are cited with source references",
            "Report is saved to the output directory",
        ],
        max_steps=30,
        invariants=[
            "Never fabricate data points or statistics",
            "Never overwrite existing files without confirmation",
        ],
    )

    # 2. Create an agent with an LLM and tools
    # The model string is resolved automatically if the API key env var is set.
    # If no API key is found, the planner falls back to heuristic mode.
    agent = Agent(
        model="claude-sonnet-4-6",
        tools=[search_sources, fetch_data, write_section, save_report],
        system_prompt="You are a senior market research analyst.",
    )

    # 3. Create a planner with MCTS search and backtracking
    planner = Planner(
        agent=agent,
        search_strategy="mcts",
        max_backtrack_depth=3,
        checkpoint_interval=2,
        num_simulations=30,
    )

    # 4. Execute the goal
    print(f"Starting goal: {goal.description}")
    print("Strategy: MCTS | Max backtrack: 3 | Checkpoints every 2 steps\n")

    result = await planner.execute(goal)

    # 5. Inspect results
    print("\nExecution complete:")
    print(f"  Success: {result.success}")
    print(f"  Steps completed: {result.steps_completed}/{result.steps_total}")
    print(f"  Goal status: {goal.status.value}")

    if result.verdicts:
        print("\nMonitor verdicts:")
        for v in result.verdicts:
            print(f"  Step {v.step_id}: {v.explanation}")

    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
