"""
Experiment-level metric aggregation for Opik.
Replaces the old experiment_metrics.py (micro confusion matrix approach).
Aggregates task-level ScoreResults into totals and averages per the metrics spec.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from opik.evaluation import test_result
from opik.evaluation.metrics import score_result

# Metrics that should be summed across tasks (raw counts)
_TOTAL_METRICS = {
    "total_tool_calls",
    "unique_actual_tool_names",
    "successful_tool_calls",
    "schema_compliant_tool_calls",
    "required_tool_names_minimal",
    "required_tool_names_optimal",
    "required_tool_calls_minimal",
    "required_tool_calls_optimal",
    "matched_tool_names_minimal",
    "matched_tool_names_optimal",
    "matched_tool_calls_minimal",
    "matched_tool_calls_optimal",
    "total_validated_criteria_minimal",
    "total_validated_criteria_optimal",
    "total_criteria_minimal",
    "total_criteria_optimal",
    "token_usage",
}

# Metrics that should be averaged across tasks (rates / scores)
_AVG_METRICS = {
    "result_accuracy_minimal",
    "result_accuracy_optimal",
    "schema_compliance_rate",
    "tool_call_success_rate",
    "precision_tool_selection_minimal",
    "precision_tool_selection_optimal",
    "recall_tool_selection_minimal",
    "recall_tool_selection_optimal",
    "f1_tool_selection_minimal",
    "f1_tool_selection_optimal",
    "precision_tool_call_minimal",
    "precision_tool_call_optimal",
    "recall_tool_call_minimal",
    "recall_tool_call_optimal",
    "f1_tool_call_minimal",
    "f1_tool_call_optimal",
    "trajectory_adherence_minimal",
    "trajectory_adherence_optimal",
    "latency_ms",
    "token_efficiency_minimal",
    "token_efficiency_optimal",
    "time_efficiency_minimal",
    "time_efficiency_optimal",
}


def compute_experiment_metrics(
    test_results: List[test_result.TestResult],
) -> List[score_result.ScoreResult]:
    """
    Aggregate task-level ScoreResults into experiment-level scores.
    - Totals: raw counts summed across all tasks
    - Averages: rates/scores averaged across all tasks
    - Failure modes: summed across tasks (higher = more tasks with that failure)
    """
    n = len(test_results)
    if n == 0:
        return []

    totals: dict[str, float] = defaultdict(float)
    avg_sums: dict[str, float] = defaultdict(float)
    avg_counts: dict[str, int] = defaultdict(int)
    failure_mode_totals: dict[str, float] = defaultdict(float)

    # Collect all known failure mode names from task results
    failure_mode_names: set[str] = set()

    for result in test_results:
        scores = {sr.name: sr.value for sr in (result.score_results or [])}

        for name, value in scores.items():
            if name in _TOTAL_METRICS:
                totals[name] += value
            elif name in _AVG_METRICS:
                avg_sums[name] += value
                avg_counts[name] += 1
            else:
                # Treat unknown metric names as potential failure modes
                failure_mode_totals[name] += value
                failure_mode_names.add(name)

    output: list[score_result.ScoreResult] = []

    for name in sorted(_TOTAL_METRICS):
        if name in totals:
            output.append(score_result.ScoreResult(
                name=f"total_{name}" if not name.startswith("total_") else name,
                value=totals[name],
                reason=f"Sum across {n} tasks",
            ))

    for name in sorted(_AVG_METRICS):
        count = avg_counts.get(name, 0)
        if count > 0:
            output.append(score_result.ScoreResult(
                name=f"avg_{name}",
                value=avg_sums[name] / count,
                reason=f"Mean across {count} tasks",
            ))

    for name in sorted(failure_mode_names):
        output.append(score_result.ScoreResult(
            name=f"failure_{name}",
            value=failure_mode_totals[name],
            reason=f"Tasks with this failure mode (out of {n})",
        ))

    return output
