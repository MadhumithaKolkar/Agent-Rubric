from agentrubric.core.trajectory import Step, Trajectory


def from_langchain(agent_output: dict, task: str) -> Trajectory:
    """
    Parse the output of a LangChain AgentExecutor.invoke() call into a Trajectory.

    agent_output is the dict returned by AgentExecutor.invoke({"input": task})
    It typically contains "input", "output", and "intermediate_steps".

    intermediate_steps is a list of (AgentAction, observation) tuples.
    """
    steps = []
    intermediate = agent_output.get("intermediate_steps", [])

    for action, observation in intermediate:
        steps.append(Step(
            step_type="tool_call",
            content="",
            tool_name=action.tool,
            tool_args={"input": action.tool_input} if isinstance(action.tool_input, str) else action.tool_input,
        ))
        steps.append(Step(
            step_type="observation",
            content=str(observation),
            tool_output=observation,
        ))

    final_answer = agent_output.get("output")
    if final_answer:
        steps.append(Step(step_type="final_answer", content=final_answer))

    return Trajectory(task=task, steps=steps, final_answer=final_answer)