from agentrubric.core.trajectory import Step, Trajectory
from agentrubric.core.report import EvaluationReport
from agentrubric.metrics.tool_accuracy import tool_accuracy
from agentrubric.metrics.argument_fidelity import argument_fidelity
from agentrubric.metrics.trajectory_length import trajectory_length
from agentrubric.metrics.completion import completion
from agentrubric.metrics.failure_taxonomy import failure_taxonomy


def evaluate(
    trajectory: Trajectory,
    expected_tools: list[str] | None = None,
    expected_calls: list[dict] | None = None,
    optimal_steps: int | None = None,
    completion_keywords: list[str] | None = None,
    tool_registry: list[str] | None = None,
) -> EvaluationReport:
    expected_tools = expected_tools or []
    expected_calls = expected_calls or []
    optimal_steps = optimal_steps or len(trajectory.steps)
    all_failures = []

    ta_score, ta_failures = tool_accuracy(trajectory, expected_tools)
    af_score, af_failures = argument_fidelity(trajectory, expected_calls)
    tl_score, tl_failures = trajectory_length(trajectory, optimal_steps)
    c_score, c_failures = completion(trajectory, completion_keywords)
    tx_failures = failure_taxonomy(trajectory, tool_registry)

    all_failures.extend(ta_failures)
    all_failures.extend(af_failures)
    all_failures.extend(tl_failures)
    all_failures.extend(c_failures)
    all_failures.extend(tx_failures)

    # deduplicate failures by type + description
    seen = set()
    unique_failures = []
    for f in all_failures:
        key = (f["type"], f["description"])
        if key not in seen:
            seen.add(key)
            unique_failures.append(f)

    overall = round(
        0.30 * ta_score +
        0.30 * af_score +
        0.20 * tl_score +
        0.20 * c_score,
        4
    )

    return EvaluationReport(
        tool_accuracy_score=ta_score,
        argument_fidelity_score=af_score,
        trajectory_length_score=tl_score,
        completion_score=c_score,
        overall_score=overall,
        failures=unique_failures,
        trajectory=trajectory,
    )


__all__ = ["Step", "Trajectory", "EvaluationReport", "evaluate"]