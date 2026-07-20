from agentrubric.core.trajectory import Trajectory


def _match_args(expected: dict, actual: dict) -> float:
    if not expected:
        return 1.0
    matches = 0
    for key, expected_val in expected.items():
        actual_val = actual.get(key)
        if actual_val is None:
            continue
        if str(actual_val).lower() == str(expected_val).lower():
            matches += 1
        elif str(expected_val).lower() in str(actual_val).lower():
            matches += 0.7
    return round(matches / len(expected), 4)


def argument_fidelity(trajectory: Trajectory, expected_calls: list[dict]) -> tuple[float, list[dict]]:
    if not expected_calls:
        return 1.0, []

    tool_call_steps = trajectory.tool_calls()
    failures = []
    scores = []

    for expected in expected_calls:
        tool_name = expected["tool_name"]
        expected_args = expected.get("expected_args", {})

        matching_steps = [s for s in tool_call_steps if s.tool_name == tool_name]

        if not matching_steps:
            failures.append({
                "type": "WRONG_TOOL",
                "step": None,
                "description": f"Expected call to '{tool_name}' but it was never called",
            })
            scores.append(0.0)
            continue

        best_score = max(_match_args(expected_args, s.tool_args or {}) for s in matching_steps)
        scores.append(best_score)

        if best_score < 0.5:
            failures.append({
                "type": "BAD_ARGS",
                "step": None,
                "description": f"Tool '{tool_name}' called with poor argument match (score {best_score:.2f})",
            })

    score = sum(scores) / len(scores) if scores else 1.0
    return round(score, 4), failures