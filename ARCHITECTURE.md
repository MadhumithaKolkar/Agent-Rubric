# Architecture

## Overview

AgentRubric evaluates LLM agent trajectories against explicit, user-defined
expectations. It does not use a model to judge quality; it computes scores
through deterministic comparison — set operations, string matching, and
arithmetic — against criteria the caller supplies.

```
Trajectory ──▶ evaluate() ──▶ EvaluationReport
```

The library is organized into four layers, each with a single responsibility:

```
agentrubric/
├── core/           data model — Trajectory, Step, EvaluationReport
├── metrics/        five independent scoring functions
├── integrations/   adapters from external agent frameworks into Trajectory
└── __init__.py     evaluate() — composes the metrics into a report
```

---

## Data model

### Step

The atomic unit of a trajectory. Every action an agent takes — thinking,
calling a tool, receiving a result, answering — is represented as one `Step`.

```python
@dataclass
class Step:
    step_type: str          # "thought" | "tool_call" | "observation" | "final_answer"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_output: Optional[Any] = None
    metadata: Optional[dict] = None
```

### Trajectory

An ordered sequence of steps plus the task that generated them.

```python
@dataclass
class Trajectory:
    task: str
    steps: list[Step]
    final_answer: Optional[str] = None
    metadata: Optional[dict] = None
```

### EvaluationReport

The result of scoring a trajectory: five sub-scores, a weighted overall
score, and a flat list of categorized failures.

```python
@dataclass
class EvaluationReport:
    tool_accuracy_score: float
    argument_fidelity_score: float
    trajectory_length_score: float
    completion_score: float
    overall_score: float
    failures: list[dict]
    trajectory: Trajectory
```

These are the only three types most consumers of the library ever touch.
Everything else is implementation detail behind `evaluate()`.

---

## Metrics

Each metric is a pure function: `(Trajectory, expectation) -> (score, failures)`.
They share no state and do not call each other, which keeps each one
independently testable and lets any subset be used in isolation.

| Metric | Input | Computation |
|---|---|---|
| Tool Accuracy | `expected_tools: list[str]` | `\|expected ∩ called\| / \|expected\|` |
| Argument Fidelity | `expected_calls: list[dict]` | per-call best-match score against actual `tool_args`, exact match = 1.0, substring match = 0.7 |
| Trajectory Length | `optimal_steps: int` | `min(1.0, optimal_steps / actual_steps)` |
| Completion | `completion_keywords: list[str]` | binary completion check, then keyword coverage of the final answer |
| Failure Taxonomy | `tool_registry: list[str]` | pattern scan for `HALLUCINATED_TOOL`, `LOOP`, `NO_COMPLETION` |

`evaluate()` (in `__init__.py`) runs all five, deduplicates the combined
failure list by `(type, description)`, and computes:

```
overall_score = 0.30 · tool_accuracy
              + 0.30 · argument_fidelity
              + 0.20 · trajectory_length
              + 0.20 · completion
```

---

## Integrations

`integrations/` contains stateless parsers that translate a specific
framework's native output into a `Trajectory`. They are the only
framework-coupled code in the library and are gated behind optional
dependency extras (`agentrubric[langchain]`, `agentrubric[openai]`) so the
core package has zero required dependencies.

- `from_langchain(agent_output, task)` — reads `AgentExecutor.invoke()`'s
  `intermediate_steps` (`(AgentAction, observation)` tuples) and the final
  `output`.
- `from_openai_messages(messages, task)` — reads a chat-completions message
  list, extracting `tool_calls` and `role: "tool"` responses.

Both produce ordinary `Trajectory` objects — from that point on, the rest of
the library has no awareness of which framework produced the data.

---

## Design decisions

**No LLM-as-judge in the core loop.** Scoring is comparison against a
caller-supplied specification, not model inference. This makes evaluation
deterministic, free, and fast enough to run on every commit in CI — a
different point on the cost/flexibility curve than open-ended quality
judging, and a deliberate one.

**Zero required dependencies.** `core/` and `metrics/` import nothing outside
the standard library. Framework support is additive via extras rather than
baked into the base install.

**Metrics are independent, not layered.** Each metric receives the raw
`Trajectory` directly rather than the output of a previous metric. This
avoids hidden coupling and lets any metric be dropped, swapped, or extended
without touching the others.

**Trajectory is the sole interchange format.** Every integration converges
on the same two dataclasses. Adding support for a new agent framework means
writing one parser function — nothing downstream changes.
