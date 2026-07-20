# AgentRubric

**Trajectory-level evaluation for LLM agents. Zero dependencies, five metrics, one score.**

> 📦 **Live on PyPI:** [pypi.org/project/agentrubric](https://pypi.org/project/agentrubric/0.1.0/) — `pip install agentrubric`

Current tools (LangSmith, Langfuse, etc.) give you logging. AgentRubric gives you scores.

When an LLM agent runs, it produces a *trajectory* — a sequence of thoughts, tool calls, observations, and a final answer. Knowing whether that trajectory was actually *correct* is hard: Did it call the right tools? With the right arguments? Did it wander in loops? Did it finish the task at all? AgentRubric answers all of that with a single `evaluate()` call.

```
trajectory (list of steps)
        │
        ▼
   AgentRubric.evaluate()
        │
        ▼
EvaluationReport
  ├─ tool_accuracy_score      — did it call the right tools?
  ├─ argument_fidelity_score  — did it call them with the right arguments?
  ├─ trajectory_length_score  — did it solve the task efficiently?
  ├─ completion_score         — did it actually finish the task?
  └─ failure_taxonomy         — what went wrong, categorised
```

---

## Install

```bash
pip install agentrubric
```

Zero required dependencies. Optional extras for framework integrations:

```bash
pip install agentrubric[langchain]   # LangChain agent parsing
pip install agentrubric[openai]      # OpenAI function-calling parsing
```

---

## Quickstart

```python
from agentrubric import Trajectory, Step, evaluate

trajectory = Trajectory(
    task="Find the latest JAX release and summarize the changelog",
    steps=[
        Step(step_type="thought", content="I need to search for JAX releases"),
        Step(step_type="tool_call", content="", tool_name="search",
             tool_args={"query": "JAX latest release"}),
        Step(step_type="observation", content="JAX 0.4.30 released on July 2026..."),
        Step(step_type="final_answer",
             content="The latest JAX release is 0.4.30, released July 2026."),
    ],
    final_answer="The latest JAX release is 0.4.30, released July 2026.",
)

report = evaluate(
    trajectory=trajectory,
    expected_tools=["search"],
    expected_calls=[{"tool_name": "search", "expected_args": {"query": "JAX latest release"}}],
    optimal_steps=4,
    completion_keywords=["JAX", "release"],
)

print(report)
print(report.overall_score)   # 0.9-ish
print(report.failures)        # []
```

```
AgentRubric Evaluation Report
========================================
  Overall Score         : 0.93
  Tool Accuracy         : 1.00
  Argument Fidelity     : 0.70
  Trajectory Efficiency : 1.00
  Completion            : 1.00

  No failures detected.
========================================
```

Export to JSON for dashboards, CI gates, or logging pipelines:

```python
report.to_json()
```

Run it yourself: `python examples/basic_evaluation.py`

---

## The five metrics

| Metric | Question it answers | Signal |
|---|---|---|
| **Tool Accuracy** | Did the agent call the tools it was supposed to? | `expected_tools` matched against tools actually called |
| **Argument Fidelity** | Were those tools called with the right arguments? | exact + fuzzy match against `expected_calls` |
| **Trajectory Length** | Did it solve the task efficiently? | `optimal_steps / actual_steps`, capped at 1.0 |
| **Completion** | Did it actually finish? | binary, or keyword coverage of the final answer |
| **Failure Taxonomy** | When it broke, why? | `WRONG_TOOL`, `MISSING_TOOL`, `BAD_ARGS`, `LOOP`, `NO_COMPLETION`, `HALLUCINATED_TOOL` |

`overall_score` is a weighted average: `0.3 × tool_accuracy + 0.3 × argument_fidelity + 0.2 × trajectory_length + 0.2 × completion`.

---

## Framework integrations

Already using LangChain or OpenAI function calling? Don't hand-build trajectories — parse them straight out of the agent's own output.

**LangChain**

```python
from agentrubric.integrations.langchain import from_langchain

result = agent_executor.invoke({"input": task})
trajectory = from_langchain(result, task=task)
report = evaluate(trajectory, expected_tools=["search"])
```

**OpenAI function calling**

```python
from agentrubric.integrations.openai import from_openai_messages

trajectory = from_openai_messages(messages, task=task)
report = evaluate(trajectory, expected_tools=["search"])
```

---

## Design principles

- **Zero required dependencies.** A library that requires nothing to install is a library people actually install. LangChain/OpenAI support is opt-in via extras.
- **Framework-agnostic core.** `Trajectory` and `Step` are plain dataclasses — build them from any agent framework, or by hand.
- **Scores, not logs.** AgentRubric doesn't replace your tracing tool; it turns a trajectory you already captured into a number you can gate CI on.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
