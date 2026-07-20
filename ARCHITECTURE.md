# AgentRubric - Architecture and Build Plan

A lightweight, framework-agnostic Python library for evaluating LLM agent trajectories.
Built for engineers who need to measure whether their agents are actually doing the right things,
not just producing text that looks right.

---

## The Problem This Solves

When an LLM agent runs, it produces a trajectory - a sequence of steps like:
- thought: "I need to search for the user's request"
- tool_call: search(query="latest JAX release")
- observation: "JAX 0.4.30 released..."
- thought: "I have the answer"
- final_answer: "The latest JAX release is 0.4.30"

Evaluating whether this trajectory is correct is hard. Did it call the right tools?
Did it call them with the right arguments? Did it take unnecessary steps? Did it fail silently?
Current tools (LangSmith, Langfuse, etc.) give you logging. They do not give you scores.
AgentRubric gives you scores.

---

## What AgentRubric Does (v0.1 scope)

Five metrics, one evaluator, one report. That is the entire v0.1.

```
trajectory (list of steps)
        |
        v
   AgentRubric.evaluate()
        |
        v
EvaluationReport
  - tool_accuracy_score      : did it call the right tools?
  - argument_fidelity_score  : did it call them with the right arguments?
  - trajectory_length_score  : did it solve the task efficiently (no unnecessary steps)?
  - completion_score         : did it actually finish the task?
  - failure_taxonomy         : what went wrong, categorised
```

---

## Folder Structure

```
agentrubric/
├── agentrubric/
│   ├── __init__.py               - public API surface (what users import)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── trajectory.py         - Trajectory and Step data models
│   │   └── report.py             - EvaluationReport data model
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── tool_accuracy.py      - metric 1: did it call the right tools?
│   │   ├── argument_fidelity.py  - metric 2: right arguments?
│   │   ├── trajectory_length.py  - metric 3: efficient path?
│   │   ├── completion.py         - metric 4: task finished?
│   │   └── failure_taxonomy.py   - metric 5: classify what went wrong
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── langchain.py          - parse LangChain agent output into Trajectory
│   │   └── openai.py             - parse OpenAI function-call output into Trajectory
│   └── report/
│       ├── __init__.py
│       └── formatter.py          - pretty-print and JSON export of EvaluationReport
├── tests/
│   ├── test_trajectory.py
│   ├── test_metrics.py
│   └── test_integrations.py
├── examples/
│   ├── basic_evaluation.py       - simplest possible usage
│   └── langchain_agent_eval.py   - real agent evaluation example
├── pyproject.toml                - packaging config (pip install agentrubric)
├── README.md                     - the public face of the project
└── ARCHITECTURE.md               - this file
```

---

## Data Models (core/)

### Step

A single step in an agent's trajectory. Everything the agent does is one of these.

```python
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class Step:
    step_type: str          # "thought", "tool_call", "observation", "final_answer"
    content: str            # the raw text content of this step
    tool_name: Optional[str] = None      # only set if step_type == "tool_call"
    tool_args: Optional[dict] = None     # only set if step_type == "tool_call"
    tool_output: Optional[Any] = None    # only set if step_type == "observation"
    metadata: Optional[dict] = None      # anything extra (latency, token count, etc.)
```

### Trajectory

A complete agent run - a sequence of steps plus task context.

```python
@dataclass
class Trajectory:
    task: str               # the original task given to the agent
    steps: list[Step]       # ordered list of every step the agent took
    final_answer: Optional[str] = None   # the agent's final response
    metadata: Optional[dict] = None
```

### EvaluationReport

The output of running AgentRubric on a trajectory.

```python
@dataclass
class EvaluationReport:
    tool_accuracy_score: float         # 0.0 to 1.0
    argument_fidelity_score: float     # 0.0 to 1.0
    trajectory_length_score: float     # 0.0 to 1.0
    completion_score: float            # 0.0 or 1.0
    overall_score: float               # weighted average of above
    failures: list[dict]               # list of {type, step, description}
    trajectory: Trajectory             # the original trajectory, stored for reference
```

---

## The Five Metrics (metrics/)

### Metric 1 - Tool Accuracy

**Question:** Did the agent call the tools it was supposed to call?

**How it works:**
- You pass in `expected_tools`: a list of tool names that should have been called
- The metric checks which of those were actually called
- Score = (number of expected tools that were called) / (total expected tools)

```python
# expected_tools = ["search", "summarize"]
# agent called: ["search", "calculator"]
# score = 1/2 = 0.5  (got search, missed summarize)
```

**File:** `metrics/tool_accuracy.py`

---

### Metric 2 - Argument Fidelity

**Question:** When the agent called a tool, did it use the right arguments?

**How it works:**
- You pass in `expected_calls`: a list of {tool_name, expected_args} dicts
- For each expected call, find the matching actual call in the trajectory
- Compare arguments using exact match for required args, fuzzy match for optional
- Score = average match rate across all expected calls

```python
# expected: search(query="JAX release notes")
# actual:   search(query="JAX release")
# partial match - query is close but not exact
# score for this call = 0.7 (configurable threshold)
```

**File:** `metrics/argument_fidelity.py`

---

### Metric 3 - Trajectory Length

**Question:** Did the agent solve the task efficiently, without unnecessary steps?

**How it works:**
- You pass in `optimal_steps`: the minimum number of steps a correct solution needs
- Score = optimal_steps / actual_steps_taken
- Capped at 1.0 (you cannot score above 1.0 for being faster than optimal)
- Steps over 2x optimal count as a failure

```python
# optimal_steps = 4
# agent took 4 steps - score = 1.0 (perfect)
# agent took 8 steps - score = 0.5 (wandered)
# agent took 12 steps - score = 0.33, flagged as failure
```

**File:** `metrics/trajectory_length.py`

---

### Metric 4 - Completion

**Question:** Did the agent actually finish the task and produce a final answer?

**How it works:**
- Binary score: 1.0 if the trajectory contains a `final_answer` step, 0.0 if not
- Optional: pass a `completion_keywords` list - words that must appear in the final answer
- If keywords are provided, score = fraction of keywords found in final answer

```python
# completion_keywords = ["price", "USD"]
# final_answer = "The price is 42 USD"
# both keywords found - score = 1.0
```

**File:** `metrics/completion.py`

---

### Metric 5 - Failure Taxonomy

**Question:** When things went wrong, what category of failure was it?

**How it works:**
- Scans the trajectory for known failure patterns
- Returns a list of labelled failures, each with a type, the step it happened at, and a description

**Failure categories:**
- `WRONG_TOOL` - called a tool not in the expected set
- `MISSING_TOOL` - never called a required tool
- `BAD_ARGS` - called the right tool with wrong arguments
- `LOOP` - called the same tool with the same args more than once (stuck in a loop)
- `NO_COMPLETION` - trajectory ended without a final answer
- `HALLUCINATED_TOOL` - called a tool that does not exist in the tool registry

```python
# failures output example:
[
    {"type": "MISSING_TOOL", "step": None, "description": "Tool 'summarize' was never called"},
    {"type": "LOOP", "step": 6, "description": "Tool 'search' called with identical args at steps 3 and 6"}
]
```

**File:** `metrics/failure_taxonomy.py`

---

## Public API (what users actually write)

The entire library is designed so that 90% of users only ever touch two things:
`Trajectory` to describe their agent run, and `evaluate()` to score it.

```python
from agentrubric import Trajectory, Step, evaluate

# Build a trajectory from your agent's run
trajectory = Trajectory(
    task="Find the latest JAX release and summarize the changelog",
    steps=[
        Step(step_type="thought", content="I need to search for JAX releases"),
        Step(step_type="tool_call", content="", tool_name="search", tool_args={"query": "JAX latest release"}),
        Step(step_type="observation", content="JAX 0.4.30 released on July 2026..."),
        Step(step_type="final_answer", content="The latest JAX release is 0.4.30, released July 2026."),
    ],
    final_answer="The latest JAX release is 0.4.30, released July 2026."
)

# Evaluate it
report = evaluate(
    trajectory=trajectory,
    expected_tools=["search"],
    expected_calls=[{"tool_name": "search", "expected_args": {"query": "JAX latest release"}}],
    optimal_steps=4,
    completion_keywords=["JAX", "release"]
)

# Read the results
print(report.overall_score)       # 0.95
print(report.failures)            # []
print(report)                     # pretty printed summary
```

---

## Integrations (integrations/)

These are thin parsers that convert existing agent framework output into AgentRubric `Trajectory` objects.
Users who already use LangChain or OpenAI do not have to manually build trajectories.

### LangChain Integration

```python
from agentrubric.integrations.langchain import from_langchain

# agent_output is what you get back from a LangChain AgentExecutor run
trajectory = from_langchain(agent_output, task="your original task")
report = evaluate(trajectory, expected_tools=["search"])
```

### OpenAI Integration

```python
from agentrubric.integrations.openai import from_openai_messages

# messages is the full messages list from an OpenAI function-calling run
trajectory = from_openai_messages(messages, task="your original task")
report = evaluate(trajectory, expected_tools=["search"])
```

---

## Build Order (do this in sequence)

**Step 1 - Core models (1-2 hours)**
Build `core/trajectory.py` and `core/report.py` first. Everything else depends on these.
Do not move to Step 2 until you can manually create a `Trajectory` with steps and print it.

**Step 2 - Metrics one by one (3-4 hours)**
Build metrics in this order: completion -> tool_accuracy -> trajectory_length -> argument_fidelity -> failure_taxonomy.
Completion is the simplest (binary). Build and test each one before moving to the next.
Write a test in `tests/test_metrics.py` for each metric as you go.

**Step 3 - The evaluate() function (1 hour)**
Wire all five metrics together in `agentrubric/__init__.py`.
The `evaluate()` function calls all five metrics and assembles an `EvaluationReport`.
Overall score = 0.3 * tool_accuracy + 0.3 * argument_fidelity + 0.2 * trajectory_length + 0.2 * completion.

**Step 4 - Report formatting (1 hour)**
Build `report/formatter.py`. Implement `__str__` on `EvaluationReport` for pretty printing.
Implement `.to_json()` for export.

**Step 5 - Examples (1 hour)**
Write `examples/basic_evaluation.py` - the simplest possible working example.
This doubles as your README's quickstart code.

**Step 6 - Integrations (2-3 hours)**
Build the LangChain parser first (more widely used), then OpenAI.
These are just data transformation functions - read the framework's output format and map it to `Step` objects.

**Step 7 - Packaging (1 hour)**
Fill in `pyproject.toml`, write the README, and run `pip install -e .` locally to confirm it works.
Then `pip install build && python -m build` to create the distribution package.

**Step 8 - PyPI release**
Create an account at pypi.org, then `pip install twine && twine upload dist/*`.
`pip install agentrubric` should work from anywhere in the world after this.

---

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "agentrubric"
version = "0.1.0"
description = "Trajectory-level evaluation for LLM agents"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
keywords = ["llm", "agents", "evaluation", "evals", "trajectory"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = []

[project.optional-dependencies]
langchain = ["langchain>=0.2.0"]
openai = ["openai>=1.0.0"]
dev = ["pytest", "black", "ruff"]

[project.urls]
Homepage = "https://github.com/MadhumithaKolkar/Agent-Rubric"
```

Note: zero required dependencies in the core library. This is intentional.
LangChain and OpenAI are optional. A library that requires nothing to install is a library people actually install.

---

## What Makes This Portfolio-Worthy

- It is a real tool that solves a real unsolved problem in production ML
- Zero dependencies in core - shows engineering discipline
- Clean data model design - shows you think in abstractions
- Framework-agnostic - shows you understand the ecosystem
- pip-installable with docs - shows you can ship, not just prototype
- The problem domain (agent evals) is exactly what GDM and every frontier lab is wrestling with in 2026

When a GDM interviewer asks "tell me about a project" - this story is:
"Agent evaluation was all vibes and logs when I built this. I designed a five-metric framework,
shipped it as an open library with zero dependencies, and it now has X downloads on PyPI."
That is a Senior MLE answer.