from agentrubric.core.trajectory import Trajectory


def completion(trajectory: Trajectory, completion_keywords: list[str] | None = None) -> tuple[float, list[dict]]:
    failures = []

    if not trajectory.completed():
        failures.append({
            "type": "NO_COMPLETION",
            "step": None,
            "description": "Agent did not produce a final answer",
        })
        return 0.0, failures

    if not completion_keywords:
        return 1.0, []

    answer = trajectory.final_answer or ""
    for step in trajectory.steps:
        if step.is_final_answer():
            answer = step.content
            break

    answer_lower = answer.lower()
    found = [kw for kw in completion_keywords if kw.lower() in answer_lower]
    score = len(found) / len(completion_keywords)

    missing = [kw for kw in completion_keywords if kw.lower() not in answer_lower]
    if missing:
        failures.append({
            "type": "INCOMPLETE_ANSWER",
            "step": None,
            "description": f"Final answer missing expected content: {missing}",
        })

    return round(score, 4), failures