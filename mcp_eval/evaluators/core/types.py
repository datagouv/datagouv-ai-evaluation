"""
Central dataclass for all per-task metric results.
Kept in a dedicated module to avoid circular imports between core modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskEvalResult:
    """All per-task metrics, ready for conversion to Opik ScoreResult objects."""

    # ── Tool usage — basics ──────────────────────────────────────────────────
    total_tool_calls: int = 0
    min_required_tool_calls_minimal: int = 0
    min_required_tool_calls_optimal: int = 0
    schema_compliant_tool_calls: int = 0
    correct_parameters_tool_calls: int = 0
    called_tool_matching_names_minimal: int = 0
    called_tool_matching_names_optimal: int = 0
    ground_truth_tool_calls_minimal: int = 0   # GT calls with correct params (minimal level)
    ground_truth_tool_calls_optimal: int = 0   # GT calls with correct params (optimal level)
    successful_tool_calls: int = 0             # calls that did not return an error

    # ── Tool usage — rates ───────────────────────────────────────────────────
    schema_compliance_rate: float = 0.0
    correct_parameters_rate: float = 0.0
    tool_call_success_rate: float = 0.0
    recall_tool_usage_minimal: float = 0.0
    recall_tool_usage_optimal: float = 0.0
    tool_call_efficiency: float = 0.0
    trajectory_adherence_minimal: float = 0.0
    trajectory_adherence_optimal: float = 0.0

    # ── Result accuracy ──────────────────────────────────────────────────────
    total_criteria_minimal: int = 0
    total_criteria_optimal: int = 0
    total_validated_criteria_minimal: int = 0
    total_validated_criteria_optimal: int = 0
    result_accuracy_minimal: float = 0.0
    result_accuracy_optimal: float = 0.0

    # ── Efficiency ───────────────────────────────────────────────────────────
    latency_ms: float = 0.0
    token_usage: int = 0
    token_efficiency_minimal: float = 0.0
    token_efficiency_optimal: float = 0.0
    time_efficiency_minimal: float = 0.0
    time_efficiency_optimal: float = 0.0

    # ── Failure modes ────────────────────────────────────────────────────────
    # {mode_name: 0|1} — 1 = failure mode is present (higher = worse)
    failure_mode_scores: dict[str, int] = field(default_factory=dict)
