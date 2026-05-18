"""
Central dataclass for all per-task metric results.
Kept in a dedicated module to avoid circular imports between core modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskEvalResult:
    """All per-task metrics, ready for conversion to Opik ScoreResult objects."""

    # ── Tool usage — actual call counts (precision denominators) ─────────────
    total_tool_calls: int = 0
    unique_actual_tool_names: int = 0
    successful_tool_calls: int = 0
    schema_compliant_tool_calls: int = 0

    # ── Tool usage — GT requirement sizes (recall denominators) ──────────────
    required_tool_names_minimal: int = 0       # |set(GT names)|
    required_tool_names_optimal: int = 0
    required_tool_calls_minimal: int = 0       # len(GT list)
    required_tool_calls_optimal: int = 0

    # ── Tool usage — matched counts (TP numerators) ──────────────────────────
    matched_tool_names_minimal: int = 0        # |set(GT names) ∩ set(actual names)|
    matched_tool_names_optimal: int = 0
    matched_tool_calls_minimal: int = 0        # LLM-judged correct calls
    matched_tool_calls_optimal: int = 0

    # ── Tool usage — rates (tool_selection level, unique names) ─────────────
    precision_tool_selection_minimal: float = 0.0
    precision_tool_selection_optimal: float = 0.0
    recall_tool_selection_minimal: float = 0.0
    recall_tool_selection_optimal: float = 0.0
    f1_tool_selection_minimal: float = 0.0
    f1_tool_selection_optimal: float = 0.0

    # ── Tool usage — rates (tool_call level, params + schema correct) ────────
    schema_compliance_rate: float = 0.0
    tool_call_success_rate: float = 0.0
    precision_tool_call_minimal: float = 0.0
    precision_tool_call_optimal: float = 0.0
    recall_tool_call_minimal: float = 0.0
    recall_tool_call_optimal: float = 0.0
    f1_tool_call_minimal: float = 0.0
    f1_tool_call_optimal: float = 0.0

    # ── Trajectory ───────────────────────────────────────────────────────────
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
