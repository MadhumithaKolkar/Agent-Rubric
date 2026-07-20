import json
from agentrubric.core.trajectory import Step, Trajectory


def from_openai_messages(messages: list[dict], task: str) -> Trajectory:
    """
    Parse an OpenAI messages list from a function-calling conversation into a Trajectory.

    messages is the list you pass to client.chat.completions.create(messages=...).
    Include the full conversation history including assistant and tool messages.
    """
    steps = []
    final_answer = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {"raw": fn.get("arguments", "")}
                    steps.append(Step(
                        step_type="tool_call",
                        content="",
                        tool_name=fn.get("name"),
                        tool_args=args,
                    ))
            elif content:
                steps.append(Step(step_type="final_answer", content=content))
                final_answer = content

        elif role == "tool":
            steps.append(Step(
                step_type="observation",
                content=content,
                tool_output=content,
            ))

    return Trajectory(task=task, steps=steps, final_answer=final_answer)