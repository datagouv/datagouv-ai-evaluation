"""
Experiment-level metric aggregation for Opik.
Replaces the old experiment_metrics.py (micro confusion matrix approach).
Aggregates task-level ScoreResults into totals and averages per the metrics spec.
"""

from collections import defaultdict
from typing import List

from opik.evaluation import test_result
from opik.evaluation.metrics import score_result

# Metrics that should be summed across tasks (raw counts)
_TOTAL_METRICS = {
    # literal tool-call track
    "total_tool_calls",
    # action track — counts
    "total_actions_made",
    "unique_action_names",
    "required_action_types_minimal",
    "required_action_types_optimal",
    "required_actions_minimal",
    "required_actions_optimal",
    "matched_action_types_minimal",
    "matched_action_types_optimal",
    "matched_actions_minimal",
    "matched_actions_optimal",
    # action track — per-capability call counts
    "calls_mcp_tool",
    "calls_datagouv_cli",
    "calls_datagouv_api_http",
    "calls_python_local_analysis",
    "calls_web_search",
    "calls_web_page_fetch",
    "calls_file_load_external",
    "calls_unclassified",
    # result accuracy
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
    "action_mapped_fraction",
    "action_success_rate",
    "precision_action_type_minimal",
    "precision_action_type_optimal",
    "recall_action_type_minimal",
    "recall_action_type_optimal",
    "f1_action_type_minimal",
    "f1_action_type_optimal",
    "precision_action_minimal",
    "precision_action_optimal",
    "recall_action_minimal",
    "recall_action_optimal",
    "f1_action_minimal",
    "f1_action_optimal",
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
            output.append(
                score_result.ScoreResult(
                    name=f"total_{name}" if not name.startswith("total_") else name,
                    value=totals[name],
                    reason=f"Sum across {n} tasks",
                )
            )

    for name in sorted(_AVG_METRICS):
        count = avg_counts.get(name, 0)
        if count > 0:
            output.append(
                score_result.ScoreResult(
                    name=f"avg_{name}",
                    value=avg_sums[name] / count,
                    reason=f"Mean across {count} tasks",
                )
            )

    for name in sorted(failure_mode_names):
        output.append(
            score_result.ScoreResult(
                name=f"failure_{name}",
                value=failure_mode_totals[name],
                reason=f"Tasks with this failure mode (out of {n})",
            )
        )

    return output
