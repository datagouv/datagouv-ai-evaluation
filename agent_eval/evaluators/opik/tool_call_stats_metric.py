"""
Opik wrapper for literal tool-call statistics.

Tracks the agent's actual tool calls (pydantic-ai ToolCallParts) — distinct from the
semantic actions they map to. `total_tool_calls` is the tool-call efficiency signal:
the same task done in fewer literal calls is cheaper. Deterministic, no LLM.
"""

from opik.evaluation.metrics import base_metric, score_result


class ToolCallStatsMetric(base_metric.BaseMetric):
    def __init__(self):
        # track=False: see ResultAccuracyMetric — avoids per-metric span that triggers
        # the dataset-compare "Avg of N trials" duplication.
        super().__init__(name="tool_call_stats", track=False)

    def score(
        self,
        input: dict,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> list[score_result.ScoreResult]:
        actual_tool_calls = (output or {}).get("actual_tool_calls") or []
        return [
            score_result.ScoreResult(
                name="total_tool_calls",
                value=float(len(actual_tool_calls)),
            ),
        ]
