"""
Efficiency metrics: latency and token usage, plus derived efficiency scores.
Pure arithmetic — no LLM calls, no Opik imports.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_eval.evaluators.core._math import safe_div


@dataclass
class EfficiencyMetrics:
    latency_ms: float
    token_usage: int
    token_efficiency_minimal: float   # result_accuracy_minimal / (token_usage / 1000)
    token_efficiency_optimal: float
    time_efficiency_minimal: float    # result_accuracy_minimal / (latency_ms / 60_000)
    time_efficiency_optimal: float


def compute_efficiency(
    latency_ms: float,
    token_usage: int,
    result_accuracy_minimal: float,
    result_accuracy_optimal: float,
) -> EfficiencyMetrics:
    """
    Derive token and time efficiency scores.
    Both are "score per unit cost" — higher is better.
    Returns 0.0 on division by zero (e.g. zero tokens or zero latency).
    """
    token_k = token_usage / 1_000  # per 1k tokens
    latency_min = latency_ms / 60_000  # per minute

    return EfficiencyMetrics(
        latency_ms=latency_ms,
        token_usage=token_usage,
        token_efficiency_minimal=safe_div(result_accuracy_minimal, token_k),
        token_efficiency_optimal=safe_div(result_accuracy_optimal, token_k),
        time_efficiency_minimal=safe_div(result_accuracy_minimal, latency_min),
        time_efficiency_optimal=safe_div(result_accuracy_optimal, latency_min),
    )
