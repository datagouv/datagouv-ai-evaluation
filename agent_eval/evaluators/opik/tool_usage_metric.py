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

from agent_eval.evaluators.core.schema_compliance import compute_schema_compliance
from agent_eval.evaluators.core.tool_params import ToolParamsOutput, judge_tool_params
from agent_eval.evaluators.core.tool_usage import (
    compute_tool_usage_basics,
    compute_tool_usage_rates,
)
from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.semantic_layer import SemanticLayerResolver
from agent_eval.tasks.loader import RequiredAction, RequiredActionArg

_SEMANTIC_LAYER_DIR = Path(__file__).parents[2] / "semantic_layer" / "config"


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


def _deserialize_required_actions(raw: list[dict]) -> list[RequiredAction]:
    actions = []
    for t in raw:
        args = [
            RequiredActionArg(
                name=a.get("name", ""),
                strict_value=a.get("strict_value"),
                criteria=a.get("criteria"),
            )
            for a in (t.get("args") or [])
        ]
        actions.append(RequiredAction(name=t["name"], args=args))
    return actions


def _resolve_to_concrete(
    actions: list[RequiredAction],
    framework: str,
    version: str,
    resolver: SemanticLayerResolver,
) -> list[RequiredAction]:
    """
    Resolve each RequiredAction's semantic name to concrete tool/command names
    and translate semantic arg names to framework-specific names.
    One semantic action with multiple concrete names expands to one entry per name.
    """
    resolved: list[RequiredAction] = []
    for action in actions:
        tool_names = resolver.resolve_tool_names(action.name, framework, version)
        if not tool_names:
            # Unknown action or no mapping — keep semantic name as-is
            tool_names = [action.name]
        for tool_name in tool_names:
            concrete_args = [
                RequiredActionArg(
                    name=resolver.resolve_arg_name(arg.name, framework, version),
                    strict_value=arg.strict_value,
                    criteria=arg.criteria,
                )
                for arg in action.args
            ]
            resolved.append(RequiredAction(name=tool_name, args=concrete_args))
    return resolved


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

        # Determine framework and version for semantic resolution
        capabilities: list[str] = out.get("capabilities") or []
        mcp_version: str = str(out.get("mcp_version") or "")
        framework = next(
            (c for c in capabilities if c in ("mcp", "web_search", "cli")),
            "mcp",  # default
        )
        resolver = SemanticLayerResolver(_SEMANTIC_LAYER_DIR)

        raw_chain = (expected_output or {}).get("action_chain", {})
        required_minimal = _resolve_to_concrete(
            _deserialize_required_actions(
                (raw_chain.get("minimal") or {}).get("required_actions") or []
            ),
            framework, mcp_version, resolver,
        )
        required_optimal = _resolve_to_concrete(
            _deserialize_required_actions(
                (raw_chain.get("optimal") or {}).get("required_actions") or []
            ),
            framework, mcp_version, resolver,
        )

        # ── Deterministic counts ─────────────────────────────────────────────
        schema_compliant, _ = compute_schema_compliance(
            actual_tool_calls, available_tools_schema
        )
        basics = compute_tool_usage_basics(
            actual_tool_calls, required_minimal, required_optimal,
            schema_compliant_count=schema_compliant,
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

        matched_calls_minimal = params_minimal.matched_tool_calls
        matched_calls_optimal = params_optimal.matched_tool_calls

        reason_minimal = _params_reason(params_minimal, actual_tool_calls)
        reason_optimal = _params_reason(params_optimal, actual_tool_calls)

        def _serialize_matches(
            matches: list,
            required: list[RequiredAction],
        ) -> list[dict]:
            result = []
            for m in matches:
                req_name = required[m.gt_index].name if m.gt_index < len(required) else None
                actual_id = actual_tool_calls[m.actual_index].get("tool_call_id") if m.actual_index < len(actual_tool_calls) else None
                result.append({
                    "tool_call_id": actual_id,
                    "actual_tool_name": m.tool_name,
                    "required_action": req_name,
                    "correct_params": m.correct,
                })
            return result

        # ── Rates ────────────────────────────────────────────────────────────
        rates = compute_tool_usage_rates(
            basics=basics,
            matched_tool_calls_minimal=matched_calls_minimal,
            matched_tool_calls_optimal=matched_calls_optimal,
        )

        return [
            # actual call counts
            score_result.ScoreResult(name="total_tool_calls", value=float(basics.total_tool_calls)),
            score_result.ScoreResult(name="unique_actual_tool_names", value=float(basics.unique_actual_tool_names)),
            score_result.ScoreResult(name="successful_tool_calls", value=float(basics.successful_tool_calls)),
            score_result.ScoreResult(name="schema_compliant_tool_calls", value=float(basics.schema_compliant_tool_calls)),
            # GT requirement sizes
            score_result.ScoreResult(name="required_tool_names_minimal", value=float(basics.required_tool_names_minimal)),
            score_result.ScoreResult(name="required_tool_names_optimal", value=float(basics.required_tool_names_optimal)),
            score_result.ScoreResult(name="required_tool_calls_minimal", value=float(basics.required_tool_calls_minimal)),
            score_result.ScoreResult(name="required_tool_calls_optimal", value=float(basics.required_tool_calls_optimal)),
            # matched counts (TP)
            score_result.ScoreResult(name="matched_tool_names_minimal", value=float(basics.matched_tool_names_minimal)),
            score_result.ScoreResult(name="matched_tool_names_optimal", value=float(basics.matched_tool_names_optimal)),
            score_result.ScoreResult(
                name="matched_tool_calls_minimal", value=float(matched_calls_minimal),
                reason=reason_minimal,
                metadata={"call_to_required_action_matches": _serialize_matches(params_minimal.matches, required_minimal)},
            ),
            score_result.ScoreResult(
                name="matched_tool_calls_optimal", value=float(matched_calls_optimal),
                reason=reason_optimal,
                metadata={"call_to_required_action_matches": _serialize_matches(params_optimal.matches, required_optimal)},
            ),
            # rates — schema / success
            score_result.ScoreResult(name="schema_compliance_rate", value=rates.schema_compliance_rate),
            score_result.ScoreResult(name="tool_call_success_rate", value=rates.tool_call_success_rate),
            # rates — tool_selection level
            score_result.ScoreResult(name="precision_tool_selection_minimal", value=rates.precision_tool_selection_minimal),
            score_result.ScoreResult(name="precision_tool_selection_optimal", value=rates.precision_tool_selection_optimal),
            score_result.ScoreResult(name="recall_tool_selection_minimal", value=rates.recall_tool_selection_minimal, reason=reason_minimal),
            score_result.ScoreResult(name="recall_tool_selection_optimal", value=rates.recall_tool_selection_optimal, reason=reason_optimal),
            score_result.ScoreResult(name="f1_tool_selection_minimal", value=rates.f1_tool_selection_minimal),
            score_result.ScoreResult(name="f1_tool_selection_optimal", value=rates.f1_tool_selection_optimal),
            # rates — tool_call level
            score_result.ScoreResult(name="precision_tool_call_minimal", value=rates.precision_tool_call_minimal, reason=reason_minimal),
            score_result.ScoreResult(name="precision_tool_call_optimal", value=rates.precision_tool_call_optimal, reason=reason_optimal),
            score_result.ScoreResult(name="recall_tool_call_minimal", value=rates.recall_tool_call_minimal, reason=reason_minimal),
            score_result.ScoreResult(name="recall_tool_call_optimal", value=rates.recall_tool_call_optimal, reason=reason_optimal),
            score_result.ScoreResult(name="f1_tool_call_minimal", value=rates.f1_tool_call_minimal),
            score_result.ScoreResult(name="f1_tool_call_optimal", value=rates.f1_tool_call_optimal),
        ]
