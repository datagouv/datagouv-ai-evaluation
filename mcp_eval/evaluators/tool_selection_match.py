import ast
import json
import math

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


class ToolSelectionMetrics(base_metric.BaseMetric):
    def __init__(self):
        super().__init__(name="tool_selection")

    def score(
        self,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> list[score_result.ScoreResult]:
        expected_calls = _parse_expected_tool_calls(
            expected_output.get("expected_tool_calls", [])
        )

        expected = set(
            _dedupe_keep_order(
                [t["name"] for t in expected_calls if "name" in t]
            )
        )
        actual = set(
            _dedupe_keep_order(
                [t["name"] for t in output.get("actual_tool_calls", []) if "name" in t]
            )
        )
        available = set(output.get("available_tool_names", []))

        tp = len(actual & expected)
        fp = len(actual - expected)
        fn = len(expected - actual)
        tn = len(available - expected - actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        total     = tp + tn + fp + fn
        accuracy  = (tp + tn) / total if total > 0 else 1.0

        mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc       = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

        return [
            score_result.ScoreResult(name="tool_precision", value=precision),
            score_result.ScoreResult(name="tool_recall",    value=recall),
            score_result.ScoreResult(name="tool_f1",        value=f1),
            score_result.ScoreResult(name="tool_jaccard",   value=jaccard),
            score_result.ScoreResult(name="tool_accuracy",  value=accuracy),
            score_result.ScoreResult(name="tool_mcc",       value=mcc),
            score_result.ScoreResult(name="tool_tp",        value=float(tp)),
            score_result.ScoreResult(name="tool_fp",        value=float(fp)),
            score_result.ScoreResult(name="tool_fn",        value=float(fn)),
            score_result.ScoreResult(name="tool_tn",        value=float(tn)),
        ]
