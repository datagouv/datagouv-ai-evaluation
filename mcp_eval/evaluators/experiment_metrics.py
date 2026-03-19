import math
from typing import List

from opik.evaluation import test_result
from opik.evaluation.metrics import score_result


def compute_micro_tool_metrics(
    test_results: List[test_result.TestResult],
) -> List[score_result.ScoreResult]:
    """Compute micro-averaged tool selection metrics across all test results.

    Uses tool_tp/fp/fn/tn ScoreResults produced by ToolSelectionMetrics per item
    to compute experiment-level micro-averages (TP/FP/FN/TN summed globally).
    Displayed in the Opik experiments table alongside macro-averages.
    """
    total_tp = total_fp = total_fn = total_tn = 0

    for result in test_results:
        scores = {sr.name: sr.value for sr in (result.score_results or [])}
        total_tp += scores.get("tool_tp", 0)
        total_fp += scores.get("tool_fp", 0)
        total_fn += scores.get("tool_fn", 0)
        total_tn += scores.get("tool_tn", 0)

    n = len(test_results)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1        = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    total = total_tp + total_tn + total_fp + total_fn
    global_accuracy = (total_tp + total_tn) / total if total > 0 else 1.0

    mcc_denom = math.sqrt(
        (total_tp + total_fp) * (total_tp + total_fn) *
        (total_tn + total_fp) * (total_tn + total_fn)
    )
    global_mcc = (total_tp * total_tn - total_fp * total_fn) / mcc_denom if mcc_denom > 0 else 0.0

    return [
        score_result.ScoreResult(
            name="micro_precision",
            value=micro_precision,
            reason=f"Micro precision across {n} items (ΣTP={total_tp}, ΣFP={total_fp})",
        ),
        score_result.ScoreResult(
            name="micro_recall",
            value=micro_recall,
            reason=f"Micro recall across {n} items (ΣTP={total_tp}, ΣFN={total_fn})",
        ),
        score_result.ScoreResult(
            name="micro_f1",
            value=micro_f1,
            reason=f"Micro F1 across {n} items",
        ),
        score_result.ScoreResult(
            name="global_accuracy",
            value=global_accuracy,
            reason=f"Global accuracy across {n} items (ΣTP={total_tp}, ΣTN={total_tn})",
        ),
        score_result.ScoreResult(
            name="global_mcc",
            value=global_mcc,
            reason=f"Global MCC across {n} items",
        ),
    ]
