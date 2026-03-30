# agent-forge

## Long-Horizon Planning Engine for LLM Agents

### The Problem

Current LLM agents are fundamentally broken at planning. Recent research (arXiv, January 2026 — "Why Reasoning Fails to Plan") proved that **reasoning does not equal planning**. Models with strong chain-of-thought reasoning fail catastrophically on long-horizon tasks due to a phenomenon called **"early myopic commitment"** — they make locally optimal choices that compound into globally terrible outcomes.

The numbers are damning:
- Performance collapses on tasks exceeding **120 steps**
- On harder task variants, failure happens in under **15 steps**
- If an agent is 85% accurate per step, a 10-step workflow succeeds only **~20% of the time**

Every major agent framework (LangGraph, CrewAI, Anthropic's Agent SDK) delegates planning to the LLM itself. Nobody has built a **dedicated planning layer** that sits between the LLM and the execution environment.

### The Solution

`agent-forge` is a standalone planning engine that wraps any LLM agent and handles long-horizon task decomposition, execution monitoring, and dynamic replanning with backtracking.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  agent-forge                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Planner │→ │ Executor │→ │  Monitor      │  │
│  │          │  │          │  │               │  │
│  │ - HTN    │  │ - Step   │  │ - Invariant   │  │
│  │   decomp │  │   runner │  │   checking    │  │
│  │ - MCTS   │  │ - Tool   │  │ - Drift       │  │
│  │   search │  │   calls  │  │   detection   │  │
│  │ - Goal   │  │ - State  │  │ - Backtrack   │  │
│  │   graphs │  │   mgmt   │  │   triggers    │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│         ↑                           │            │
│         └───── Replan Loop ─────────┘            │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │           State Manager                   │    │
│  │  - World state tracking                   │    │
│  │  - Checkpoint / rollback                  │    │
│  │  - Causal dependency graph                │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
          │                    │
          ▼                    ▼
   ┌────────────┐      ┌────────────┐
   │  Any LLM   │      │  Any Tools │
   │  (via API) │      │  (MCP/fn)  │
   └────────────┘      └────────────┘
```

### Core Components

#### 1. Hierarchical Task Network (HTN) Planner
- Decomposes high-level goals into subtask trees
- Each subtask has preconditions, effects, and invariants
- The LLM proposes decompositions; the planner validates them against constraints
- Supports partial-order planning (parallelizable subtasks identified automatically)

#### 2. Monte Carlo Tree Search (MCTS) for Plan Selection
- When multiple plan paths exist, uses MCTS to simulate outcomes
- Lightweight rollouts using a smaller/faster model (e.g., Haiku) to estimate plan viability
- Balances exploration vs. exploitation across plan branches

#### 3. Execution Monitor with Backtracking
- After each step, checks:
  - Did the postcondition hold?
  - Are global invariants still satisfied?
  - Has the world state drifted from expectations?
- If a check fails, triggers **backtracking** to the last valid checkpoint
- Replans from that checkpoint with updated world knowledge

#### 4. State Manager
- Maintains a **causal dependency graph** of all completed actions
- Supports checkpoint/rollback (like git for agent state)
- Tracks which future plan steps are invalidated by a failed step
- Enables "what-if" analysis for plan alternatives

### Key Design Decisions

- **Model-agnostic**: Works with any LLM via standard API (OpenAI, Anthropic, local models). The planner doesn't replace the LLM — it constrains and guides it.
- **MCP-native**: Tool calls go through MCP, so any MCP server is automatically available.
- **Separation of concerns**: The LLM handles creativity/reasoning. The planner handles structure/validation. The monitor handles safety/correctness.
- **Lightweight**: Not a framework — a library. `pip install agent-forge`, wrap your existing agent, done.

### Technical Stack

- **Language**: Python 3.11+
- **Core deps**: `pydantic` (state schemas), `networkx` (dependency graphs), `httpx` (async API calls)
- **Optional**: `jax` or `numpy` for MCTS simulations
- **LLM interface**: OpenAI-compatible API, Anthropic SDK, or any callable
- **Tool interface**: MCP protocol, or plain Python functions

### API Surface (Draft)

```python
from agent_forge import Planner, Agent, Goal

# Define a goal with success criteria
goal = Goal(
    description="Research and write a comprehensive market analysis report",
    success_criteria=[
        "Report contains data from at least 5 sources",
        "All claims are cited",
        "Report is saved to output directory"
    ],
    max_steps=50,
    invariants=[
        "Never fabricate data points",
        "Never overwrite existing files without confirmation"
    ]
)

# Wrap any LLM-based agent
agent = Agent(
    model="claude-sonnet-4-6",
    tools=[...],  # MCP servers or Python functions
)

# Create a planner and execute
planner = Planner(
    agent=agent,
    search_strategy="mcts",     # or "greedy", "beam"
    max_backtrack_depth=5,
    checkpoint_interval=3,       # checkpoint every 3 steps
    rollout_model="claude-haiku-4-5",  # cheap model for plan simulation
)

result = await planner.execute(goal)
print(result.plan_trace)     # full execution tree
print(result.backtrack_log)  # where it had to retry
print(result.success)        # bool
```

### What Makes This Novel

1. **First dedicated planning layer for LLM agents** — everyone else just prompts harder
2. **Grounded in real research** — directly addresses the "reasoning ≠ planning" findings
3. **MCTS for plan selection** — borrowed from game AI (AlphaGo), never applied to LLM agent planning
4. **Backtracking with causal reasoning** — knows which downstream steps are invalidated when something fails
5. **Quantifiable improvement** — can benchmark against vanilla agents on standard tasks (SWE-bench, GAIA, WebArena)

### Benchmarking Plan

Test on established agent benchmarks:
- **GAIA** (complex real-world queries) — current SOTA: 61%
- **SWE-bench** (code editing tasks) — measure multi-file task success
- **WebArena** (web navigation) — long-horizon browser tasks
- **Custom long-horizon suite** — 50+ step tasks designed to break vanilla agents

Report: steps-to-completion, backtrack frequency, success rate vs. vanilla agent, token cost overhead.

### Repo Structure

```
agent-forge/
├── README.md
├── pyproject.toml
├── src/
│   └── agent_forge/
│       ├── __init__.py
│       ├── planner.py          # HTN decomposition + MCTS
│       ├── executor.py         # Step-by-step execution
│       ├── monitor.py          # Invariant checking + drift detection
│       ├── state.py            # State manager + checkpointing
│       ├── backtrack.py        # Backtracking engine
│       ├── goal.py             # Goal definition + success criteria
│       ├── models/
│       │   ├── openai.py       # OpenAI-compatible interface
│       │   ├── anthropic.py    # Anthropic interface
│       │   └── base.py         # Abstract model interface
│       └── tools/
│           ├── mcp.py          # MCP tool integration
│           └── function.py     # Plain Python function tools
├── tests/
│   ├── test_planner.py
│   ├── test_backtrack.py
│   ├── test_mcts.py
│   └── benchmarks/
│       ├── gaia.py
│       └── long_horizon.py
├── examples/
│   ├── research_agent.py
│   ├── coding_agent.py
│   └── web_agent.py
└── docs/
    ├── architecture.md
    ├── benchmarks.md
    └── api.md
```

### Research References

- "Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents" (arXiv:2601.22311, Jan 2026)
- "DeepPlanning: Benchmarking Long-Horizon Agentic Planning" (arXiv:2601.18137, Jan 2026)
- "Hierarchical Task Network Planning with LLMs" (NeurIPS 2025 Workshop)
- "Monte Carlo Tree Search for Language Model Decoding" (ICML 2025)
- GAIA Benchmark: current SOTA 61% (Writer's Action Agent)
