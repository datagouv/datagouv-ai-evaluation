"""
Opik wrapper for trajectory adherence evaluation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from opik.evaluation.metrics import base_metric, score_result

from mcp_eval.evaluators.core.trajectory import compute_trajectory_adherence
from mcp_eval.evaluators.core.judge_model import JudgeModel
from mcp_eval.tasks.loader import (
    RequiredTool,
    RequiredToolArg,
    ToolChain,
    ToolChainLevel,
)


def _deserialize_level(raw: dict) -> ToolChainLevel:
    tools = [
        RequiredTool(
            name=t["name"],
            args=[
                RequiredToolArg(
                    name=a.get("name", ""),
                    strict_value=a.get("strict_value"),
                    criteria=a.get("criteria"),
                )
                for a in (t.get("args") or [])
            ],
        )
        for t in (raw.get("required_tools") or [])
    ]
    return ToolChainLevel(chain=raw.get("chain") or "", required_tools=tools)


class TrajectoryAdherenceMetric(base_metric.BaseMetric):
    def __init__(self, judge_model_path: Path):
        super().__init__(name="trajectory_adherence")
        self._judge_model = JudgeModel(judge_model_path)

    def score(
        self,
        input: dict,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> list[score_result.ScoreResult]:
        user_prompt = (input or {}).get("prompt", "")
        actual_tool_calls = (output or {}).get("actual_tool_calls") or []

        raw_chain = (expected_output or {}).get("tool_chain", {})
        tool_chain = ToolChain(
            minimal=_deserialize_level(raw_chain.get("minimal") or {}),
            optimal=_deserialize_level(raw_chain.get("optimal") or {}),
        )

        result = asyncio.run(
            compute_trajectory_adherence(
                model=self._judge_model,
                tool_chain=tool_chain,
                actual_tool_calls=actual_tool_calls,
                user_prompt=user_prompt,
            )
        )

        return [
            score_result.ScoreResult(
                name="trajectory_adherence_minimal",
                value=result.score_minimal,
                reason=result.explanation_minimal,
            ),
            score_result.ScoreResult(
                name="trajectory_adherence_optimal",
                value=result.score_optimal,
                reason=result.explanation_optimal,
            ),
        ]
