"""
Central dataclass for all per-task metric results.
Kept in a dedicated module to avoid circular imports between core modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskEvalResult:
    """All per-task metrics, ready for conversion to Opik ScoreResult objects."""

    # ── Tool calls — literal agent invocations (efficiency tracking) ─────────
    total_tool_calls: int = 0

    # ── Action usage — actual counts (precision denominators) ────────────────
    total_actions_made: int = 0                # len(mapped action instances)
    unique_action_names: int = 0               # |set(mapped action names)|
    action_success_rate: float = 0.0           # instances whose source call did not error
    action_mapped_fraction: float = 0.0        # fraction of calls mapped to >= 1 action

    # ── Action usage — GT requirement sizes (recall denominators) ────────────
    required_action_types_minimal: int = 0     # |set(GT names)|
    required_action_types_optimal: int = 0
    required_actions_minimal: int = 0          # len(GT list)
    required_actions_optimal: int = 0

    # ── Action usage — matched counts (TP numerators) ────────────────────────
    matched_action_types_minimal: int = 0      # |set(GT names) ∩ set(actual names)|
    matched_action_types_optimal: int = 0
    matched_actions_minimal: int = 0           # LLM-judged correct actions
    matched_actions_optimal: int = 0

    # ── Action usage — rates (action-type level, unique names) ──────────────
    precision_action_type_minimal: float = 0.0
    precision_action_type_optimal: float = 0.0
    recall_action_type_minimal: float = 0.0
    recall_action_type_optimal: float = 0.0
    f1_action_type_minimal: float = 0.0
    f1_action_type_optimal: float = 0.0

    # ── Action usage — rates (action-instance level, params correct) ─────────
    precision_action_minimal: float = 0.0
    precision_action_optimal: float = 0.0
    recall_action_minimal: float = 0.0
    recall_action_optimal: float = 0.0
    f1_action_minimal: float = 0.0
    f1_action_optimal: float = 0.0

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
