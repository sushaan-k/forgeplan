"""Coding agent example -- multi-file code generation with validation.

Demonstrates using agent-forge for a coding task where:
- The agent must generate code across multiple files.
- Each file must pass syntax checks (postconditions).
- Invariants ensure existing files are not deleted.
- Backtracking recovers from failed test runs.

Requirements:
    pip install agent-forge
    export OPENAI_API_KEY="sk-..." or ANTHROPIC_API_KEY="sk-..."
"""

from __future__ import annotations

import ast
import asyncio

from agent_forge import Agent, Goal, Planner

# --- Tool definitions ---


def write_code(filename: str, code: str) -> str:
    """Write Python code to a file (simulated)."""
    # In production, this would write to the filesystem
    return f"Wrote {len(code)} chars to {filename}"


def check_syntax(code: str) -> str:
    """Check Python code for syntax errors."""
    try:
        ast.parse(code)
        return "Syntax OK"
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"


def run_tests(test_file: str) -> str:
    """Run pytest on a test file (simulated)."""
    # In production, this would invoke pytest subprocess
    return f"All tests passed in {test_file}"


def list_files(directory: str) -> str:
    """List files in a directory (simulated)."""
    return f"Files in {directory}: __init__.py, models.py, utils.py"


async def main() -> None:
    """Run the coding agent."""
    goal = Goal(
        description=(
            "Implement a Python REST API with models, routes, and tests. "
            "The API should have a User model with CRUD endpoints."
        ),
        success_criteria=[
            "models.py contains a User dataclass",
            "routes.py contains CRUD endpoint handlers",
            "test_api.py contains tests for all endpoints",
            "All code passes syntax checks",
        ],
        max_steps=25,
        invariants=[
            "Never delete existing source files",
            "All generated code must be valid Python",
        ],
    )

    agent = Agent(
        model="gpt-4o",
        tools=[write_code, check_syntax, run_tests, list_files],
        system_prompt="You are a senior Python backend engineer.",
    )

    # Use beam search for coding tasks -- faster than MCTS and
    # still evaluates multiple candidate plans.
    planner = Planner(
        agent=agent,
        search_strategy="beam",
        max_backtrack_depth=4,
        checkpoint_interval=3,
    )

    print(f"Goal: {goal.description}")
    print("Strategy: beam | Max backtrack: 4\n")

    result = await planner.execute(goal)

    print("\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")
    print(f"  Final state keys: {list(result.final_state.keys())}")

    if result.error:
        print(f"  Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
