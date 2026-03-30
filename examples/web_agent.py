"""Web navigation agent example -- long-horizon browsing with recovery.

Demonstrates agent-forge on a web navigation task where:
- Steps can fail (page not found, timeout, element not visible).
- Checkpoints let the agent recover mid-session.
- Invariants prevent navigating to disallowed domains.
- MCTS selects the most promising navigation strategy.

Requirements:
    pip install agent-forge
    export ANTHROPIC_API_KEY="sk-..." or OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import asyncio

from agent_forge import Agent, Goal, Planner

# --- Simulated browser tools ---


def navigate(url: str) -> str:
    """Navigate to a URL in the browser."""
    return f"Navigated to {url} -- page loaded (title: 'Example Page')"


def click_element(selector: str) -> str:
    """Click an element on the current page."""
    return f"Clicked element matching '{selector}'"


def fill_form(selector: str, value: str) -> str:
    """Fill a form field with a value."""
    return f"Filled '{selector}' with '{value}'"


def extract_text(selector: str) -> str:
    """Extract text content from an element."""
    return f"Extracted text from '{selector}': 'Sample extracted content'"


def take_screenshot(filename: str) -> str:
    """Take a screenshot of the current page."""
    return f"Screenshot saved to {filename}"


async def main() -> None:
    """Run the web navigation agent."""
    goal = Goal(
        description=(
            "Navigate to the company's investor relations page, "
            "find the latest quarterly earnings report, "
            "extract key financial metrics (revenue, net income, EPS), "
            "and compile them into a summary."
        ),
        success_criteria=[
            "Revenue figure extracted and recorded",
            "Net income figure extracted and recorded",
            "EPS figure extracted and recorded",
            "Summary saved with all three metrics",
        ],
        max_steps=40,
        invariants=[
            "Never navigate to domains outside the target company site",
            "Never submit forms without user confirmation",
            "Never store credentials in plain text",
        ],
    )

    agent = Agent(
        model="claude-sonnet-4-6",
        tools=[navigate, click_element, fill_form, extract_text, take_screenshot],
        system_prompt=(
            "You are a web navigation agent. You interact with web pages "
            "through browser tools. Be methodical and verify each step."
        ),
    )

    # MCTS is ideal for web navigation -- many possible paths,
    # need to evaluate which sequence of clicks is most likely
    # to reach the target content.
    planner = Planner(
        agent=agent,
        search_strategy="mcts",
        max_backtrack_depth=5,
        checkpoint_interval=2,
        num_simulations=40,
    )

    print(f"Goal: {goal.description[:80]}...")
    print("Strategy: MCTS | Max backtrack: 5 | Simulations: 40\n")

    result = await planner.execute(goal)

    print("\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")
    print(f"  Verdicts: {len(result.verdicts)}")

    # Show backtrack events if any occurred
    passed = sum(1 for v in result.verdicts if not v.should_backtrack)
    failed = sum(1 for v in result.verdicts if v.should_backtrack)
    print(f"  Monitoring: {passed} passed, {failed} triggered backtrack")

    if result.error:
        print(f"  Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
