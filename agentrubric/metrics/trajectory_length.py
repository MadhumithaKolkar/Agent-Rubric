from agentrubric.core.trajectory import Trajectory


def trajectory_length(trajectory: Trajectory, optimal_steps: int) -> tuple[float, list[dict]]:
    if optimal_steps <= 0:
        return 1.0, []

    actual = len(trajectory.steps)
    failures = []

    if actual == 0:
        return 0.0, [{"type": "NO_COMPLETION", "step": None, "description": "Trajectory has no steps"}]

    score = min(1.0, optimal_steps / actual)

    if actual > optimal_steps * 2:
        failures.append({
            "type": "INEFFICIENT_TRAJECTORY",
            "step": actual,
            "description": f"Agent took {actual} steps, optimal is {optimal_steps} (over 2x threshold)",
        })

    return round(score, 4), failures