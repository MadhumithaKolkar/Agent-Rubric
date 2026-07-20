from agentrubric.core.trajectory import Trajectory


def failure_taxonomy(trajectory: Trajectory, tool_registry: list[str] | None = None) -> list[dict]:
    failures = []
    tool_calls = trajectory.tool_calls()

    # detect hallucinated tools - called a tool not in the registry
    if tool_registry:
        registry_set = set(tool_registry)
        for i, step in enumerate(tool_calls):
            if step.tool_name and step.tool_name not in registry_set:
                failures.append({
                    "type": "HALLUCINATED_TOOL",
                    "step": i,
                    "description": f"Tool '{step.tool_name}' does not exist in the tool registry",
                })

    # detect loops - same tool called with identical args more than once
    seen_calls = {}
    for i, step in enumerate(tool_calls):
        if not step.tool_name:
            continue
        key = (step.tool_name, str(sorted((step.tool_args or {}).items())))
        if key in seen_calls:
            failures.append({
                "type": "LOOP",
                "step": i,
                "description": (
                    f"Tool '{step.tool_name}' called with identical args "
                    f"at steps {seen_calls[key]} and {i}"
                ),
            })
        else:
            seen_calls[key] = i

    # detect no completion
    if not trajectory.completed():
        failures.append({
            "type": "NO_COMPLETION",
            "step": None,
            "description": "Trajectory ended without a final answer",
        })

    return failures