from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class Step:
    step_type: str
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_output: Optional[Any] = None
    metadata: Optional[dict] = None

    def is_tool_call(self) -> bool:
        return self.step_type == "tool_call"

    def is_final_answer(self) -> bool:
        return self.step_type == "final_answer"


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    metadata: Optional[dict] = None

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.is_tool_call()]

    def tools_called(self) -> list[str]:
        return [s.tool_name for s in self.tool_calls() if s.tool_name]

    def completed(self) -> bool:
        return self.final_answer is not None or any(s.is_final_answer() for s in self.steps)