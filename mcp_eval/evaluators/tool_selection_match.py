import ast
import json

from opik.evaluation.metrics import base_metric, score_result


def _parse_expected_tool_calls(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        try:
            return json.loads(value)
        except Exception:
            return ast.literal_eval(value)
    return []


def _dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


class ToolSelectionMatch(base_metric.BaseMetric):
    def __init__(self):
        super().__init__(name="tool_selection_match")

    def score(
        self,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> score_result.ScoreResult:
        expected_calls = _parse_expected_tool_calls(
            expected_output.get("expected_tool_calls", [])
        )

        expected_tool_names = _dedupe_keep_order(
            [tool["name"] for tool in expected_calls if "name" in tool]
        )
        actual_tool_names = _dedupe_keep_order(
            [tool["name"] for tool in output.get("actual_tool_calls", []) if "name" in tool]
        )

        if not expected_tool_names:
            value = 1.0
            reason = "no expected tools"
        else:
            correct = [n for n in actual_tool_names if n in expected_tool_names]
            value = len(correct) / len(expected_tool_names)
            reason = f"matched {len(correct)}/{len(expected_tool_names)} expected tools"

        return score_result.ScoreResult(name=self.name, value=value, reason=reason)
