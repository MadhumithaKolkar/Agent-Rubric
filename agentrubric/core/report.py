from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentrubric.core.trajectory import Trajectory


@dataclass
class EvaluationReport:
    tool_accuracy_score: float
    argument_fidelity_score: float
    trajectory_length_score: float
    completion_score: float
    overall_score: float
    failures: list[dict] = field(default_factory=list)
    trajectory: "Trajectory" = None

    def __str__(self) -> str:
        lines = [
            "",
            "AgentRubric Evaluation Report",
            "=" * 40,
            f"  Overall Score         : {self.overall_score:.2f}",
            f"  Tool Accuracy         : {self.tool_accuracy_score:.2f}",
            f"  Argument Fidelity     : {self.argument_fidelity_score:.2f}",
            f"  Trajectory Efficiency : {self.trajectory_length_score:.2f}",
            f"  Completion            : {self.completion_score:.2f}",
        ]
        if self.failures:
            lines.append("")
            lines.append(f"  Failures ({len(self.failures)}):")
            for f in self.failures:
                step_str = f"step {f['step']}" if f.get("step") else "n/a"
                lines.append(f"    [{f['type']}] {f['description']} ({step_str})")
        else:
            lines.append("")
            lines.append("  No failures detected.")
        lines.append("=" * 40)
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "tool_accuracy_score": self.tool_accuracy_score,
            "argument_fidelity_score": self.argument_fidelity_score,
            "trajectory_length_score": self.trajectory_length_score,
            "completion_score": self.completion_score,
            "failures": self.failures,
        }