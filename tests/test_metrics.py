import pytest
from agentrubric.core.trajectory import Step, Trajectory
from agentrubric.metrics.tool_accuracy import tool_accuracy
from agentrubric.metrics.argument_fidelity import argument_fidelity
from agentrubric.metrics.trajectory_length import trajectory_length
from agentrubric.metrics.completion import completion
from agentrubric.metrics.failure_taxonomy import failure_taxonomy


def make_trajectory(tool_names=None, has_final_answer=True):
    steps = []
    for name in (tool_names or []):
        steps.append(Step(step_type="tool_call", content="", tool_name=name, tool_args={"query": "test"}))
    if has_final_answer:
        steps.append(Step(step_type="final_answer", content="The answer is 42"))
    return Trajectory(
        task="test task",
        steps=steps,
        final_answer="The answer is 42" if has_final_answer else None,
    )


class TestToolAccuracy:
    def test_perfect_score(self):
        t = make_trajectory(["search", "summarize"])
        score, failures = tool_accuracy(t, ["search", "summarize"])
        assert score == 1.0
        assert failures == []

    def test_missing_tool(self):
        t = make_trajectory(["search"])
        score, failures = tool_accuracy(t, ["search", "summarize"])
        assert score == 0.5
        assert any(f["type"] == "MISSING_TOOL" for f in failures)

    def test_no_expected_tools(self):
        t = make_trajectory(["search"])
        score, failures = tool_accuracy(t, [])
        assert score == 1.0


class TestTrajectoryLength:
    def test_optimal(self):
        t = make_trajectory(["search"])
        score, failures = trajectory_length(t, optimal_steps=2)
        assert score == 1.0

    def test_over_threshold(self):
        t = make_trajectory(["a", "b", "c", "d", "e"])
        score, failures = trajectory_length(t, optimal_steps=2)
        assert score < 0.4
        assert any(f["type"] == "INEFFICIENT_TRAJECTORY" for f in failures)


class TestCompletion:
    def test_completed_with_keywords(self):
        t = make_trajectory(["search"])
        score, failures = completion(t, completion_keywords=["answer", "42"])
        assert score == 1.0

    def test_missing_keyword(self):
        t = make_trajectory(["search"])
        score, failures = completion(t, completion_keywords=["answer", "missing_word"])
        assert score == 0.5

    def test_no_completion(self):
        t = make_trajectory(["search"], has_final_answer=False)
        score, failures = completion(t)
        assert score == 0.0
        assert any(f["type"] == "NO_COMPLETION" for f in failures)


class TestFailureTaxonomy:
    def test_detects_loop(self):
        steps = [
            Step(step_type="tool_call", content="", tool_name="search", tool_args={"query": "jax"}),
            Step(step_type="tool_call", content="", tool_name="search", tool_args={"query": "jax"}),
        ]
        t = Trajectory(task="test", steps=steps)
        failures = failure_taxonomy(t)
        assert any(f["type"] == "LOOP" for f in failures)

    def test_detects_hallucinated_tool(self):
        steps = [
            Step(step_type="tool_call", content="", tool_name="fake_tool", tool_args={}),
        ]
        t = Trajectory(task="test", steps=steps)
        failures = failure_taxonomy(t, tool_registry=["search", "summarize"])
        assert any(f["type"] == "HALLUCINATED_TOOL" for f in failures)