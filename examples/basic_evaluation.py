from agentrubric import Step, Trajectory, evaluate

# Build a trajectory from a hypothetical agent run
# Task: find the latest JAX release and summarize the changelog
trajectory = Trajectory(
    task="Find the latest JAX release and summarize the changelog",
    steps=[
        Step(step_type="thought", content="I need to search for the latest JAX release"),
        Step(
            step_type="tool_call",
            content="",
            tool_name="search",
            tool_args={"query": "JAX latest release changelog"},
        ),
        Step(
            step_type="observation",
            content="JAX 0.4.30 was released in July 2026. Changes include improved TPU support...",
            tool_output="JAX 0.4.30 released July 2026...",
        ),
        Step(
            step_type="final_answer",
            content="The latest JAX release is 0.4.30, released July 2026. Key changes include improved TPU support.",
        ),
    ],
    final_answer="The latest JAX release is 0.4.30, released July 2026.",
)

# Evaluate the trajectory
report = evaluate(
    trajectory=trajectory,
    expected_tools=["search"],
    expected_calls=[
        {"tool_name": "search", "expected_args": {"query": "JAX latest release"}}
    ],
    optimal_steps=4,
    completion_keywords=["JAX", "0.4.30"],
    tool_registry=["search", "summarize", "calculator"],
)

print(report)
print("\nJSON export:")
import json
print(json.dumps(report.to_json(), indent=2))