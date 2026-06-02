"""Shared arithmetic helpers for evaluator metrics."""
from __future__ import annotations


def safe_div(a: float, b: float) -> float:
    return round(a / b, 6) if b > 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if (precision + recall) > 0 else 0.0
