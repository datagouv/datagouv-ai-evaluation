"""
Opik wrapper for action call mapping.

Runs before ToolUsageMetric and TrajectoryAdherenceMetric.
Classifies each actual_tool_call by capability category and maps it to a semantic action.

Returns:
  - value: fraction of tool calls mapped to a known semantic action
  - metadata: per-call mappings + per-category counts
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from opik.evaluation.metrics import base_metric, score_result

from agent_eval.evaluators.core.action_mapper import map_action_calls
from agent_eval.evaluators.core.judge_model import JudgeModel

_SEMANTIC_LAYER_DIR = Path(__file__).parents[2] / "semantic_layer" / "config"


class ActionCallMapperMetric(base_metric.BaseMetric):
    def __init__(self, judge_model_path: Path):
        super().__init__(name="action_call_mapper")
        self._judge_model = JudgeModel(judge_model_path)

    def score(
        self,
        input: dict,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> list[score_result.ScoreResult]:
        out = output or {}
        actual_tool_calls = out.get("actual_tool_calls") or []
        available_tool_names = out.get("available_tool_names") or []
        mcp_version = str(out.get("mcp_version") or "")

        mapper_output = asyncio.run(
            map_action_calls(
                actual_tool_calls=actual_tool_calls,
                available_tool_names=available_tool_names,
                mcp_version=mcp_version,
                semantic_layer_dir=_SEMANTIC_LAYER_DIR,
                judge_model=self._judge_model,
            )
        )

        counts = mapper_output.counts
        mappings_serializable = [
            {
                "tool_call_id": m.tool_call_id,
                "tool_name": m.tool_name,
                "capability_category": m.capability_category,
                "semantic_actions": m.semantic_actions,
                "criteria_pending": [
                    {"action": a, "criteria": c} for a, c in m.criteria_pending
                ],
                "confidence": m.confidence,
                "reason": m.reason,
            }
            for m in mapper_output.mappings
        ]

        n_mapped = sum(1 for m in mapper_output.mappings if m.semantic_actions)
        return [
            score_result.ScoreResult(
                name="action_mapped_fraction",
                value=mapper_output.mapped_fraction,
                reason=f"Mapped {n_mapped} of {len(mapper_output.mappings)} tool calls to semantic actions",
                metadata={
                    "mappings": mappings_serializable,
                    "counts": counts,
                },
            ),
            # Per-category counts as individual scores for easy aggregation
            score_result.ScoreResult(
                name="calls_mcp_tool",
                value=float(counts.get("mcp_tool", 0)),
            ),
            score_result.ScoreResult(
                name="calls_datagouv_cli",
                value=float(counts.get("datagouv_cli", 0)),
            ),
            score_result.ScoreResult(
                name="calls_datagouv_api_http",
                value=float(counts.get("datagouv_api_http", 0)),
            ),
            score_result.ScoreResult(
                name="calls_python_local_analysis",
                value=float(counts.get("python_local_analysis", 0)),
            ),
            score_result.ScoreResult(
                name="calls_web_search",
                value=float(counts.get("web_search", 0)),
            ),
            score_result.ScoreResult(
                name="calls_web_page_fetch",
                value=float(counts.get("web_page_fetch", 0)),
            ),
            score_result.ScoreResult(
                name="calls_file_load_external",
                value=float(counts.get("file_load_external", 0)),
            ),
            score_result.ScoreResult(
                name="calls_unclassified",
                value=float(counts.get("unclassified", 0)),
            ),
        ]
