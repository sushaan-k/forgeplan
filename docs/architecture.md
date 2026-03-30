# Architecture

## Overview

agent-forge is structured around five core components that form a closed planning-execution-monitoring loop. The design separates concerns so that the LLM handles creativity and reasoning while the planning engine handles structure, validation, and recovery.

## Component Diagram

```
                         +-----------+
                         |   Goal    |
                         | (user)    |
                         +-----+-----+
                               |
                               v
+------------------------------+------------------------------+
|                          Planner                             |
|                                                              |
|  1. HTN Decomposition: Goal -> candidate subtask trees       |
|  2. Plan Selection: MCTS / Beam / Greedy                     |
|  3. Replan Loop: retry on backtrack with updated state       |
+-----+------------------------------------------+------------+
      |                                          ^
      v                                          | backtrack
+-----+------+        +-----------+        +-----+-------+
|  Executor  | -----> |  Monitor  | -----> | Backtrack   |
|            |        |           |        |   Engine    |
| - run step |        | - post-   |        | - find CP   |
| - call tool|        |   cond.   |        | - invalidate|
| - call LLM |        | - invar.  |        | - rollback  |
| - update   |        | - drift   |        |             |
|   state    |        |           |        |             |
+-----+------+        +-----+-----+        +-------------+
      |                      |
      v                      v
+-----+----------------------+-----+
|        State Manager              |
|                                   |
| - world state (versioned dict)    |
| - checkpoints (snapshot/restore)  |
| - causal dependency graph (DAG)   |
| - step history                    |
+-----------------------------------+
```

## Data Flow

1. The user defines a **Goal** with a description, success criteria, invariants, and step limits.

2. The **Planner** sends the goal to the LLM via an HTN decomposition prompt. The LLM returns 1-3 candidate plans, each a sequence of steps with actions, preconditions, and postconditions.

3. If multiple plans exist, the **MCTS search** (or beam/greedy) evaluates them through simulated rollouts and selects the best candidate.

4. The **Executor** runs each step:
   - If the step's action matches a registered tool, the tool is invoked.
   - Otherwise, the LLM generates the step's output.
   - The step's `expected_state_changes` are applied to the state manager.
   - A checkpoint is created every N steps.

5. After each step, the **Monitor** evaluates:
   - **Postcondition**: did the step achieve what it claimed?
   - **Invariants**: are global safety constraints still satisfied?
   - **Drift**: has the actual state diverged from the plan's expectations?

6. If any check fails, the **BacktrackEngine**:
   - Finds the most recent valid checkpoint.
   - Rolls back world state.
   - Invalidates all causally-dependent downstream steps.
   - Returns control to the planner for replanning.

7. The planner re-decomposes the goal from the restored state and tries again, up to `max_backtrack_depth` times.

## Key Design Decisions

### Why HTN over flat plan generation?

Hierarchical Task Networks let us validate plans at multiple levels of abstraction. A high-level step like "research the market" decomposes into concrete subtasks, each with checkable preconditions and postconditions. This is more robust than asking the LLM for a flat list of steps.

### Why MCTS for plan selection?

When the LLM generates multiple candidate plans, we need a principled way to choose. MCTS (borrowed from AlphaGo and game AI) balances exploration vs. exploitation across plan branches. Lightweight rollouts using a cheaper model estimate each plan's probability of success.

### Why causal dependency graphs?

When a step fails, naive backtracking throws away all subsequent work. By tracking which steps causally depend on the failed step, we can invalidate only the affected downstream steps and preserve valid work on independent branches.

### Why checkpoints instead of full undo?

Full undo logging is expensive and complex. Periodic checkpoints (like database snapshots) provide a good tradeoff: we lose at most `checkpoint_interval` steps of work, but the implementation is simple and the state manager stays fast.

## Extension Points

- **Custom model backends**: Implement `BaseModel.generate()` to support any LLM.
- **Custom tools**: Pass any Python callable or MCP server URL.
- **Custom monitors**: Subclass `Monitor` to add domain-specific checks.
- **Custom search**: The `MCTSSearch` class can be replaced with any strategy that selects from `list[list[PlanStep]]`.
