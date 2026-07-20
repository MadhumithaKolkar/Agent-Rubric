from agentrubric.core.trajectory import Trajectory


def tool_accuracy(trajectory: Trajectory, expected_tools: list[str]) -> tuple[float, list[dict]]:
    if not expected_tools:
        return 1.0, []

    called = set(trajectory.tools_called())
    expected = set(expected_tools)
    failures = []

    for tool in expected:
        if tool not in called:
            failures.append({
                "type": "MISSING_TOOL",
                "step": None,
                "description": f"Tool '{tool}' was expected but never called",
            })

    score = len(expected & called) / len(expected)
    return round(score, 4), failures