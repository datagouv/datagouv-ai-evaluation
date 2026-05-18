"""
Deterministic tool-usage metrics.
All computation here is pure Python — no LLM calls, no Opik imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_eval.tasks.loader import RequiredTool


@dataclass
class ToolUsageBasics:
    # actual call counts (precision denominators)
    total_tool_calls: int
    unique_actual_tool_names: int               # |set(actual names)|
    successful_tool_calls: int
    schema_compliant_tool_calls: int
    # GT requirement sizes (recall denominators)
    required_tool_names_minimal: int            # |set(GT names)|, minimal level
    required_tool_names_optimal: int
    required_tool_calls_minimal: int            # len(GT list), minimal level
    required_tool_calls_optimal: int
    # matched counts (TP numerators)
    matched_tool_names_minimal: int             # |set(GT names) ∩ set(actual names)|
    matched_tool_names_optimal: int


@dataclass
class ToolUsageRates:
    schema_compliance_rate: float
    tool_call_success_rate: float
    # tool_selection level (unique name sets)
    precision_tool_selection_minimal: float
    precision_tool_selection_optimal: float
    recall_tool_selection_minimal: float
    recall_tool_selection_optimal: float
    f1_tool_selection_minimal: float
    f1_tool_selection_optimal: float
    # tool_call level (multiset, name + params + schema)
    precision_tool_call_minimal: float
    precision_tool_call_optimal: float
    recall_tool_call_minimal: float
    recall_tool_call_optimal: float
    f1_tool_call_minimal: float
    f1_tool_call_optimal: float


def _is_error(result: Any) -> bool:
    """Return True if a tool call result looks like an error."""
    if result is None:
        return False
    s = str(result).lower()
    return s.startswith("error") or "exception" in s or "traceback" in s


def compute_tool_usage_basics(
    actual_tool_calls: list[dict[str, Any]],
    required_tools_minimal: list[RequiredTool],
    required_tools_optimal: list[RequiredTool],
    schema_compliant_count: int = 0,
) -> ToolUsageBasics:
    """
    Compute deterministic tool-usage counts.

    actual_tool_calls: list of dicts with keys "name", "arguments", "result"
    required_tools_minimal: ground-truth tools for the minimal level
    required_tools_optimal: ground-truth tools for the optimal level
    schema_compliant_count: number of calls passing schema validation (computed externally)
    """
    actual_names = {c.get("name", "") for c in actual_tool_calls}
    gt_names_minimal = {t.name for t in required_tools_minimal}
    gt_names_optimal = {t.name for t in required_tools_optimal}

    return ToolUsageBasics(
        total_tool_calls=len(actual_tool_calls),
        unique_actual_tool_names=len(actual_names),
        successful_tool_calls=sum(1 for c in actual_tool_calls if not _is_error(c.get("result"))),
        schema_compliant_tool_calls=schema_compliant_count,
        required_tool_names_minimal=len(gt_names_minimal),
        required_tool_names_optimal=len(gt_names_optimal),
        required_tool_calls_minimal=len(required_tools_minimal),
        required_tool_calls_optimal=len(required_tools_optimal),
        matched_tool_names_minimal=len(gt_names_minimal & actual_names),
        matched_tool_names_optimal=len(gt_names_optimal & actual_names),
    )


def compute_tool_usage_rates(
    basics: ToolUsageBasics,
    matched_tool_calls_minimal: int,
    matched_tool_calls_optimal: int,
) -> ToolUsageRates:
    """
    Derive rate metrics from basics + LLM-judged matched_tool_calls counts.
    All rates are [0, 1] and guard against division by zero (return 0.0).
    """
    def safe_div(a: float, b: float) -> float:
        return round(a / b, 6) if b > 0 else 0.0

    def f1(p: float, r: float) -> float:
        return round(2 * p * r / (p + r), 6) if (p + r) > 0 else 0.0

    p_sel_min = safe_div(basics.matched_tool_names_minimal, basics.unique_actual_tool_names)
    p_sel_opt = safe_div(basics.matched_tool_names_optimal, basics.unique_actual_tool_names)
    r_sel_min = safe_div(basics.matched_tool_names_minimal, basics.required_tool_names_minimal)
    r_sel_opt = safe_div(basics.matched_tool_names_optimal, basics.required_tool_names_optimal)

    p_call_min = safe_div(matched_tool_calls_minimal, basics.total_tool_calls)
    p_call_opt = safe_div(matched_tool_calls_optimal, basics.total_tool_calls)
    r_call_min = safe_div(matched_tool_calls_minimal, basics.required_tool_calls_minimal)
    r_call_opt = safe_div(matched_tool_calls_optimal, basics.required_tool_calls_optimal)

    return ToolUsageRates(
        schema_compliance_rate=safe_div(basics.schema_compliant_tool_calls, basics.total_tool_calls),
        tool_call_success_rate=safe_div(basics.successful_tool_calls, basics.total_tool_calls),
        precision_tool_selection_minimal=p_sel_min,
        precision_tool_selection_optimal=p_sel_opt,
        recall_tool_selection_minimal=r_sel_min,
        recall_tool_selection_optimal=r_sel_opt,
        f1_tool_selection_minimal=f1(p_sel_min, r_sel_min),
        f1_tool_selection_optimal=f1(p_sel_opt, r_sel_opt),
        precision_tool_call_minimal=p_call_min,
        precision_tool_call_optimal=p_call_opt,
        recall_tool_call_minimal=r_call_min,
        recall_tool_call_optimal=r_call_opt,
        f1_tool_call_minimal=f1(p_call_min, r_call_min),
        f1_tool_call_optimal=f1(p_call_opt, r_call_opt),
    )
