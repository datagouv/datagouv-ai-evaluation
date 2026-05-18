"""
Opik wrapper for tool-usage metrics.
Combines: tool_usage (deterministic), schema_compliance, tool_params (LLM judge).
Efficiency metrics live in ResultAccuracyMetric.
Trajectory is handled separately in TrajectoryAdherenceMetric.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from opik.evaluation.metrics import base_metric, score_result

from mcp_eval.evaluators.core.schema_compliance import compute_schema_compliance
from mcp_eval.evaluators.core.tool_params import ToolParamsOutput, judge_tool_params
from mcp_eval.evaluators.core.tool_usage import (
    compute_tool_usage_basics,
    compute_tool_usage_rates,
)
from mcp_eval.evaluators.core.judge_model import JudgeModel
from mcp_eval.tasks.loader import RequiredTool, RequiredToolArg


def _params_reason(output: ToolParamsOutput, actual_tool_calls: list[dict]) -> str:
    lines = []
    for m in output.matches:
        marker = "✓" if m.validated else "✗"
        actual_args = ""
        if m.actual_index < len(actual_tool_calls):
            args = (
                actual_tool_calls[m.actual_index].get("arguments")
                or actual_tool_calls[m.actual_index].get("args")
                or {}
            )
            if args:
                actual_args = f" args={str(args)[:120]}"
        lines.append(f"[{marker}] {m.tool_name}{actual_args}: {m.explanation}")
    if not lines:
        return "No tool parameter matches evaluated."
    return "\n".join(lines)


def _deserialize_required_tools(raw: list[dict]) -> list[RequiredTool]:
    tools = []
    for t in raw:
        args = [
            RequiredToolArg(
                name=a.get("name", ""),
                strict_value=a.get("strict_value"),
                criteria=a.get("criteria"),
            )
            for a in (t.get("args") or [])
        ]
        tools.append(RequiredTool(name=t["name"], args=args))
    return tools


class ToolUsageMetric(base_metric.BaseMetric):
    def __init__(self, judge_model_path: Path):
        super().__init__(name="tool_usage")
        self._judge_model = JudgeModel(judge_model_path)

    def score(
        self,
        input: dict,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> list[score_result.ScoreResult]:
        user_prompt = (input or {}).get("prompt", "")
        out = output or {}

        actual_tool_calls: list[dict[str, Any]] = out.get("actual_tool_calls") or []
        available_tools_schema: list[dict] = out.get("available_tools_schema") or []

        raw_chain = (expected_output or {}).get("tool_chain", {})
        required_minimal = _deserialize_required_tools(
            (raw_chain.get("minimal") or {}).get("required_tools") or []
        )
        required_optimal = _deserialize_required_tools(
            (raw_chain.get("optimal") or {}).get("required_tools") or []
        )

        # ── Deterministic counts ─────────────────────────────────────────────
        basics = compute_tool_usage_basics(
            actual_tool_calls, required_minimal, required_optimal
        )
        schema_compliant, _ = compute_schema_compliance(
            actual_tool_calls, available_tools_schema
        )

        # ── LLM judge — run both levels concurrently ─────────────────────────
        async def _run_judges():
            return await asyncio.gather(
                judge_tool_params(
                    self._judge_model,
                    user_prompt,
                    actual_tool_calls,
                    required_minimal,
                    available_tools_schema,
                ),
                judge_tool_params(
                    self._judge_model,
                    user_prompt,
                    actual_tool_calls,
                    required_optimal,
                    available_tools_schema,
                ),
            )

        params_minimal, params_optimal = asyncio.run(_run_judges())

        gt_minimal = params_minimal.ground_truth_tool_calls
        gt_optimal = params_optimal.ground_truth_tool_calls
        # correct_parameters uses the higher-coverage level (optimal superset)
        correct_params = params_optimal.correct_parameters_tool_calls

        reason_minimal = _params_reason(params_minimal, actual_tool_calls)
        reason_optimal = _params_reason(params_optimal, actual_tool_calls)

        # ── Rates ────────────────────────────────────────────────────────────
        rates = compute_tool_usage_rates(
            basics=basics,
            schema_compliant_tool_calls=schema_compliant,
            correct_parameters_tool_calls=correct_params,
            ground_truth_tool_calls_minimal=gt_minimal,
            ground_truth_tool_calls_optimal=gt_optimal,
        )

        return [
            # basics
            score_result.ScoreResult(
                name="total_tool_calls", value=float(basics.total_tool_calls)
            ),
            score_result.ScoreResult(
                name="min_required_tool_calls_minimal",
                value=float(basics.min_required_tool_calls_minimal),
            ),
            score_result.ScoreResult(
                name="min_required_tool_calls_optimal",
                value=float(basics.min_required_tool_calls_optimal),
            ),
            score_result.ScoreResult(
                name="schema_compliant_tool_calls", value=float(schema_compliant)
            ),
            score_result.ScoreResult(
                name="correct_parameters_tool_calls",
                value=float(correct_params),
                reason=reason_optimal,
            ),
            score_result.ScoreResult(
                name="called_tool_matching_names_minimal",
                value=float(basics.called_tool_matching_names_minimal),
            ),
            score_result.ScoreResult(
                name="called_tool_matching_names_optimal",
                value=float(basics.called_tool_matching_names_optimal),
            ),
            score_result.ScoreResult(
                name="ground_truth_tool_calls_minimal",
                value=float(gt_minimal),
                reason=reason_minimal,
            ),
            score_result.ScoreResult(
                name="ground_truth_tool_calls_optimal",
                value=float(gt_optimal),
                reason=reason_optimal,
            ),
            score_result.ScoreResult(
                name="successful_tool_calls", value=float(basics.successful_tool_calls)
            ),
            # rates
            score_result.ScoreResult(
                name="schema_compliance_rate", value=rates.schema_compliance_rate
            ),
            score_result.ScoreResult(
                name="correct_parameters_rate",
                value=rates.correct_parameters_rate,
                reason=reason_optimal,
            ),
            score_result.ScoreResult(
                name="tool_call_success_rate", value=rates.tool_call_success_rate
            ),
            score_result.ScoreResult(
                name="recall_tool_usage_minimal",
                value=rates.recall_tool_usage_minimal,
                reason=reason_minimal,
            ),
            score_result.ScoreResult(
                name="recall_tool_usage_optimal",
                value=rates.recall_tool_usage_optimal,
                reason=reason_optimal,
            ),
            score_result.ScoreResult(
                name="tool_call_efficiency", value=rates.tool_call_efficiency
            ),
        ]
