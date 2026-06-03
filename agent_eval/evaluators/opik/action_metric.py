"""
Opik wrapper for action-based evaluation (semantic space).

Runs the action mapper ONCE (the single source of truth) to turn the agent's literal
tool calls into a flat sequence of semantic action instances, then scores:
  - action mapping / observability (mapped fraction, per-capability counts, success rate)
  - action usage (precision/recall/F1 at action-type and action-instance levels)
  - action parameter correctness (feeds the action-instance matched counts)
  - trajectory adherence (sequence alignment of the mapped actions)

Everything below the mapper sees the same instances, so usage/params/trajectory are consistent
and the mapper LLM pass runs only once.
"""

import asyncio
from pathlib import Path

from opik.evaluation.metrics import base_metric, score_result

from agent_eval.evaluators.core.action_mapper import map_action_calls
from agent_eval.evaluators.core.action_params import (
    ActionParamsOutput,
    judge_action_params,
)
from agent_eval.evaluators.core.action_usage import (
    compute_action_usage_basics,
    compute_action_usage_rates,
)
from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.evaluators.core.trajectory import compute_trajectory_adherence
from agent_eval.tasks.loader import (
    ActionChain,
    ActionChainLevel,
    RequiredAction,
    RequiredActionArg,
)

_SEMANTIC_LAYER_DIR = Path(__file__).parents[2] / "semantic_layer" / "config"

_CATEGORY_SCORES = [
    "mcp_tool",
    "datagouv_cli",
    "datagouv_api_http",
    "python_local_analysis",
    "web_search",
    "web_page_fetch",
    "file_load_external",
    "unclassified",
]


def _deserialize_level(raw: dict) -> ActionChainLevel:
    actions = [
        RequiredAction(
            name=t["name"],
            args=[
                RequiredActionArg(
                    name=a.get("name", ""),
                    strict_value=a.get("strict_value"),
                    criteria=a.get("criteria"),
                )
                for a in (t.get("args") or [])
            ],
        )
        for t in (raw.get("required_actions") or [])
    ]
    return ActionChainLevel(chain=raw.get("chain") or "", required_actions=actions)


def _params_reason(output: ActionParamsOutput) -> str:
    lines = []
    for m in output.matches:
        marker = "✓" if m.validated else "✗"
        lines.append(f"[{marker}] {m.action}: {m.explanation}")
    if not lines:
        return "No action parameter matches evaluated."
    return "\n".join(lines)


def _serialize_matches(output: ActionParamsOutput) -> list[dict]:
    return [
        {
            "action": m.action,
            "tool_call_id": m.source_tool_call_id,
            "correct_params": m.correct,
        }
        for m in output.matches
    ]


class ActionMetric(base_metric.BaseMetric):
    def __init__(self, judge_model_path: Path):
        # track=False: see ResultAccuracyMetric — avoids per-metric span that triggers
        # the dataset-compare "Avg of N trials" duplication.
        super().__init__(name="action", track=False)
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

        actual_tool_calls = out.get("actual_tool_calls") or []
        available_tool_names = out.get("available_tool_names") or []
        mcp_version = str(out.get("mcp_version") or "")

        raw_chain = (expected_output or {}).get("action_chain", {})
        action_chain = ActionChain(
            minimal=_deserialize_level(raw_chain.get("minimal") or {}),
            optimal=_deserialize_level(raw_chain.get("optimal") or {}),
        )
        required_minimal = action_chain.minimal.required_actions
        required_optimal = action_chain.optimal.required_actions

        async def _run_all():
            mapper_output = await map_action_calls(
                actual_tool_calls=actual_tool_calls,
                available_tool_names=available_tool_names,
                mcp_version=mcp_version,
                semantic_layer_dir=_SEMANTIC_LAYER_DIR,
                judge_model=self._judge_model,
            )
            instances = mapper_output.instances
            params_minimal, params_optimal, trajectory = await asyncio.gather(
                judge_action_params(
                    self._judge_model, user_prompt, instances, required_minimal
                ),
                judge_action_params(
                    self._judge_model, user_prompt, instances, required_optimal
                ),
                compute_trajectory_adherence(
                    self._judge_model, action_chain, instances, user_prompt
                ),
            )
            return mapper_output, params_minimal, params_optimal, trajectory

        mapper_output, params_minimal, params_optimal, trajectory = asyncio.run(
            _run_all()
        )

        instances = mapper_output.instances
        basics = compute_action_usage_basics(
            instances, required_minimal, required_optimal
        )
        matched_minimal = params_minimal.matched_actions
        matched_optimal = params_optimal.matched_actions
        rates = compute_action_usage_rates(basics, matched_minimal, matched_optimal)

        reason_minimal = _params_reason(params_minimal)
        reason_optimal = _params_reason(params_optimal)
        counts = mapper_output.counts

        call_mappings_serializable = [
            {
                "tool_call_id": cm.tool_call_id,
                "tool_name": cm.tool_name,
                "capability_category": cm.capability_category,
                "errored": cm.errored,
                "actions": [{"action": a.action, "args": a.args} for a in cm.actions],
                "confidence": cm.confidence,
                "reason": cm.reason,
            }
            for cm in mapper_output.call_mappings
        ]
        n_mapped = sum(1 for cm in mapper_output.call_mappings if cm.actions)

        return [
            # ── mapping / observability ──────────────────────────────────────
            score_result.ScoreResult(
                name="action_mapped_fraction",
                value=mapper_output.mapped_fraction,
                reason=f"Mapped {n_mapped} of {len(mapper_output.call_mappings)} tool calls to semantic actions",
                metadata={
                    "call_mappings": call_mappings_serializable,
                    "counts": counts,
                },
            ),
            score_result.ScoreResult(
                name="total_actions_made", value=float(basics.total_actions_made)
            ),
            score_result.ScoreResult(
                name="unique_action_names", value=float(basics.unique_action_names)
            ),
            score_result.ScoreResult(
                name="action_success_rate", value=basics.action_success_rate
            ),
            *[
                score_result.ScoreResult(
                    name=f"calls_{cat}", value=float(counts.get(cat, 0))
                )
                for cat in _CATEGORY_SCORES
            ],
            # ── GT requirement sizes ─────────────────────────────────────────
            score_result.ScoreResult(
                name="required_action_types_minimal",
                value=float(basics.required_action_types_minimal),
            ),
            score_result.ScoreResult(
                name="required_action_types_optimal",
                value=float(basics.required_action_types_optimal),
            ),
            score_result.ScoreResult(
                name="required_actions_minimal",
                value=float(basics.required_actions_minimal),
            ),
            score_result.ScoreResult(
                name="required_actions_optimal",
                value=float(basics.required_actions_optimal),
            ),
            # ── matched counts (TP) ──────────────────────────────────────────
            score_result.ScoreResult(
                name="matched_action_types_minimal",
                value=float(basics.matched_action_types_minimal),
            ),
            score_result.ScoreResult(
                name="matched_action_types_optimal",
                value=float(basics.matched_action_types_optimal),
            ),
            score_result.ScoreResult(
                name="matched_actions_minimal",
                value=float(matched_minimal),
                reason=reason_minimal,
                metadata={
                    "action_to_required_matches": _serialize_matches(params_minimal)
                },
            ),
            score_result.ScoreResult(
                name="matched_actions_optimal",
                value=float(matched_optimal),
                reason=reason_optimal,
                metadata={
                    "action_to_required_matches": _serialize_matches(params_optimal)
                },
            ),
            # ── rates — action-type level ────────────────────────────────────
            score_result.ScoreResult(
                name="precision_action_type_minimal",
                value=rates.precision_action_type_minimal,
            ),
            score_result.ScoreResult(
                name="precision_action_type_optimal",
                value=rates.precision_action_type_optimal,
            ),
            score_result.ScoreResult(
                name="recall_action_type_minimal",
                value=rates.recall_action_type_minimal,
            ),
            score_result.ScoreResult(
                name="recall_action_type_optimal",
                value=rates.recall_action_type_optimal,
            ),
            score_result.ScoreResult(
                name="f1_action_type_minimal", value=rates.f1_action_type_minimal
            ),
            score_result.ScoreResult(
                name="f1_action_type_optimal", value=rates.f1_action_type_optimal
            ),
            # ── rates — action-instance level ────────────────────────────────
            score_result.ScoreResult(
                name="precision_action_minimal",
                value=rates.precision_action_minimal,
                reason=reason_minimal,
            ),
            score_result.ScoreResult(
                name="precision_action_optimal",
                value=rates.precision_action_optimal,
                reason=reason_optimal,
            ),
            score_result.ScoreResult(
                name="recall_action_minimal",
                value=rates.recall_action_minimal,
                reason=reason_minimal,
            ),
            score_result.ScoreResult(
                name="recall_action_optimal",
                value=rates.recall_action_optimal,
                reason=reason_optimal,
            ),
            score_result.ScoreResult(
                name="f1_action_minimal", value=rates.f1_action_minimal
            ),
            score_result.ScoreResult(
                name="f1_action_optimal", value=rates.f1_action_optimal
            ),
            # ── trajectory ───────────────────────────────────────────────────
            score_result.ScoreResult(
                name="trajectory_adherence_minimal",
                value=trajectory.score_minimal,
                reason=trajectory.explanation_minimal,
            ),
            score_result.ScoreResult(
                name="trajectory_adherence_optimal",
                value=trajectory.score_optimal,
                reason=trajectory.explanation_optimal,
            ),
        ]
