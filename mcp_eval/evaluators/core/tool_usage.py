"""
Deterministic tool-usage metrics.
All computation here is pure Python — no LLM calls, no Opik imports.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from mcp_eval.tasks.loader import RequiredTool


@dataclass
class ToolUsageBasics:
    total_tool_calls: int
    min_required_tool_calls_minimal: int
    min_required_tool_calls_optimal: int
    called_tool_matching_names_minimal: int
    called_tool_matching_names_optimal: int
    successful_tool_calls: int


@dataclass
class ToolUsageRates:
    schema_compliance_rate: float
    correct_parameters_rate: float
    tool_call_success_rate: float
    recall_tool_usage_minimal: float
    recall_tool_usage_optimal: float
    tool_call_efficiency: float


def _is_error(result: Any) -> bool:
    """Return True if a tool call result looks like an error."""
    if result is None:
        return False
    s = str(result).lower()
    return s.startswith("error") or "exception" in s or "traceback" in s


def _count_name_matches(
    required_tools: list[RequiredTool],
    actual_tool_calls: list[dict[str, Any]],
) -> int:
    """
    Count how many ground-truth calls (by name) are covered by actual calls.
    Handles duplicates: if the same tool appears N times in GT, up to N actual calls
    of that name are counted (min(GT count, actual count) per tool name).
    This reflects "basic tool selection" — name matching only, no arg inspection.
    """
    gt_counts = Counter(t.name for t in required_tools)
    actual_counts = Counter(c.get("name", "") for c in actual_tool_calls)
    return sum(min(gt_counts[name], actual_counts[name]) for name in gt_counts)


def compute_tool_usage_basics(
    actual_tool_calls: list[dict[str, Any]],
    required_tools_minimal: list[RequiredTool],
    required_tools_optimal: list[RequiredTool],
) -> ToolUsageBasics:
    """
    Compute deterministic tool-usage counts.

    actual_tool_calls: list of dicts with keys "name", "arguments", "result"
    required_tools_minimal: ground-truth tools for the minimal level
    required_tools_optimal: ground-truth tools for the optimal level (minimal + optimal combined)
    """
    total = len(actual_tool_calls)
    successful = sum(1 for c in actual_tool_calls if not _is_error(c.get("result")))

    return ToolUsageBasics(
        total_tool_calls=total,
        min_required_tool_calls_minimal=len(required_tools_minimal),
        min_required_tool_calls_optimal=len(required_tools_optimal),
        called_tool_matching_names_minimal=_count_name_matches(required_tools_minimal, actual_tool_calls),
        called_tool_matching_names_optimal=_count_name_matches(required_tools_optimal, actual_tool_calls),
        successful_tool_calls=successful,
    )


def compute_tool_usage_rates(
    basics: ToolUsageBasics,
    schema_compliant_tool_calls: int,
    correct_parameters_tool_calls: int,
    ground_truth_tool_calls_minimal: int,
    ground_truth_tool_calls_optimal: int,
) -> ToolUsageRates:
    """
    Derive rate metrics from basics + LLM-judged counts.
    All rates are [0, 1] and guard against division by zero (return 0.0).
    """
    total = basics.total_tool_calls

    def safe_div(a: float, b: float) -> float:
        return round(a / b, 6) if b > 0 else 0.0

    # tool_call_efficiency = ground_truth calls / total calls
    # Use the average of minimal and optimal GT counts so the rate stays in [0, 1]
    avg_gt = (ground_truth_tool_calls_minimal + ground_truth_tool_calls_optimal) / 2

    return ToolUsageRates(
        schema_compliance_rate=safe_div(schema_compliant_tool_calls, total),
        correct_parameters_rate=safe_div(correct_parameters_tool_calls, total),
        tool_call_success_rate=safe_div(basics.successful_tool_calls, total),
        recall_tool_usage_minimal=safe_div(
            ground_truth_tool_calls_minimal, basics.min_required_tool_calls_minimal
        ),
        recall_tool_usage_optimal=safe_div(
            ground_truth_tool_calls_optimal, basics.min_required_tool_calls_optimal
        ),
        tool_call_efficiency=safe_div(avg_gt, total),
    )
