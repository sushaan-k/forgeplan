# API Reference

## Core Classes

### `Goal`

```python
from agent_forge import Goal, GoalStatus
```

High-level objective for the planning engine.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `description` | `str` | required | Natural-language description of the goal |
| `success_criteria` | `list[str]` | `[]` | Conditions that must hold for success |
| `max_steps` | `int` | `100` | Maximum execution steps |
| `invariants` | `list[str]` | `[]` | Conditions that must never be violated |
| `priority` | `int` | `0` | Priority level (higher = more important) |
| `metadata` | `dict[str, str]` | `{}` | Arbitrary key-value context |

**Methods:**

- `mark_in_progress()` -- transition to IN_PROGRESS
- `mark_completed()` -- transition to COMPLETED
- `mark_failed()` -- transition to FAILED
- `mark_cancelled()` -- transition to CANCELLED

---

### `Agent`

```python
from agent_forge import Agent
```

Wraps an LLM and tools into a configuration the Planner can drive.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| BaseModel \| None` | `None` | Model name string or instance |
| `tools` | `list[Any]` | `[]` | Tool objects or raw callables |
| `system_prompt` | `str` | `""` | System prompt for the LLM |

**Properties:**

- `model` -- the resolved LLM backend (or None)
- `function_tools` -- all tools as FunctionTool instances

**Model resolution:** When `model` is a string like `"claude-sonnet-4-6"` or `"gpt-4o"`, the Agent checks for the corresponding API key environment variable (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) and creates the appropriate backend. If no key is found, `model` is None and the planner uses heuristic mode.

---

### `Planner`

```python
from agent_forge import Planner
```

Main entry point. Decomposes goals, selects plans, and executes with monitoring.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent` | `Agent` | required | Agent wrapping the LLM and tools |
| `search_strategy` | `str \| SearchStrategy` | `"mcts"` | `"greedy"`, `"mcts"`, or `"beam"` |
| `max_backtrack_depth` | `int` | `5` | Max consecutive backtracks |
| `checkpoint_interval` | `int` | `3` | Steps between checkpoints |
| `rollout_model` | `BaseModel \| None` | `None` | Cheaper model for MCTS rollouts |
| `num_simulations` | `int` | `50` | MCTS simulation count |

**Methods:**

- `async execute(goal: Goal) -> ExecutionResult` -- plan and execute a goal

---

### `ExecutionResult`

```python
from agent_forge import ExecutionResult
```

Returned by `Planner.execute()`.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the plan completed successfully |
| `steps_completed` | `int` | Steps that ran to completion |
| `steps_total` | `int` | Total steps in the plan |
| `verdicts` | `list[MonitorVerdict]` | Per-step monitoring results |
| `final_state` | `dict[str, Any]` | World state at end of execution |
| `error` | `str` | Error message if failed |

---

### `PlanStep`

```python
from agent_forge import PlanStep
```

A single executable step within a plan.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier |
| `description` | `str` | What this step should accomplish |
| `action` | `str` | Tool name or "llm" |
| `action_args` | `dict` | Arguments for the action |
| `preconditions` | `list[str]` | Must hold before execution |
| `postcondition` | `str` | Should hold after execution |
| `expected_state_changes` | `dict` | Predicted state updates |
| `depends_on` | `list[str]` | IDs of prerequisite steps |
| `subtasks` | `list[PlanStep]` | Nested sub-steps (HTN) |
| `status` | `StepStatus` | Current execution status |
| `result` | `Any` | Output after execution |

---

## State Management

### `StateManager`

```python
from agent_forge import StateManager
```

Manages world state, checkpoints, and causal dependencies.

**Key methods:**

- `get(key, default=None)` -- retrieve a state value
- `set(key, value)` -- update a state value
- `update(dict)` -- batch update
- `advance_step()` -- increment step counter
- `create_checkpoint()` -- snapshot current state
- `rollback_to_checkpoint(id)` -- restore from checkpoint
- `add_dependency(from_step, to_step)` -- record causal dependency
- `get_dependents(step_id)` -- find transitive dependents
- `invalidate_step(step_id)` -- invalidate step and dependents

---

## Monitoring

### `Monitor`

```python
from agent_forge import Monitor
```

Evaluates postconditions, invariants, and drift after each step.

**Key methods:**

- `async check_postcondition(step_id, postcondition, result)` -- check a postcondition
- `async check_invariants(step_id, invariants)` -- check all invariants
- `compute_drift(expected_state)` -- measure state drift (0.0 to 1.0)
- `async evaluate_step(...)` -- run all checks, return MonitorVerdict

### `MonitorVerdict`

| Field | Type | Description |
|---|---|---|
| `step_id` | `str` | Step that was monitored |
| `postcondition_result` | `CheckResult` | PASSED, FAILED, or UNCERTAIN |
| `invariant_results` | `dict[str, CheckResult]` | Per-invariant results |
| `drift_score` | `float` | State drift (0.0 = no drift) |
| `should_backtrack` | `bool` | Whether to trigger backtracking |
| `explanation` | `str` | Human-readable summary |

---

## Backtracking

### `BacktrackEngine`

```python
from agent_forge import BacktrackEngine
```

Manages checkpoint-based backtracking with causal awareness.

**Key methods:**

- `can_backtrack()` -- check if backtracking is possible
- `backtrack(failed_step_id, reason)` -- execute backtrack
- `reset_depth()` -- reset consecutive backtrack counter
- `find_best_checkpoint(step_id)` -- find optimal checkpoint
- `get_summary()` -- backtracking statistics

---

## Tools

### `FunctionTool`

```python
from agent_forge.tools import FunctionTool
```

Wraps a Python callable as an agent tool.

```python
def search(query: str, limit: int = 10) -> str:
    return f"Results for {query}"

tool = FunctionTool(fn=search, name="search", description="Search the web")
result = await tool.execute(query="AI planning")
schema = tool.to_schema()  # OpenAI-compatible format
```

### `MCPTool`

```python
from agent_forge.tools import MCPTool
```

Client for MCP (Model Context Protocol) server tools.

```python
# Discover all tools on an MCP server
tools = await MCPTool.discover("http://localhost:3000")

# Invoke a specific tool
tool = MCPTool(server_url="http://localhost:3000", tool_name="search")
result = await tool.execute(query="market data")
```

---

## Model Backends

### `BaseModel` (abstract)

```python
from agent_forge.models.base import BaseModel, ModelResponse
```

All model backends implement this interface.

**Abstract methods:**

- `async generate(messages, tools=None, **kwargs) -> ModelResponse`

**Convenience methods:**

- `async generate_text(prompt, system=None) -> str`

### `OpenAIModel`

```python
from agent_forge.models.openai import OpenAIModel
```

Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API.

### `AnthropicModel`

```python
from agent_forge.models.anthropic import AnthropicModel
```

Works with Anthropic's Claude models via the Messages API.

---

## Exceptions

All exceptions inherit from `AgentForgeError`.

| Exception | When |
|---|---|
| `PlanningError` | Planner fails to generate a valid plan |
| `ExecutionError` | A plan step fails during execution |
| `BacktrackError` | Backtracking fails or exceeds max depth |
| `InvariantViolation` | A global invariant is violated |
| `StateError` | Checkpoint/rollback operation fails |
| `ModelError` | LLM API call fails |
| `ToolError` | Tool invocation fails |
