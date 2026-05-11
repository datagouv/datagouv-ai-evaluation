"""
LLM-as-a-judge for trajectory adherence (sequence alignment).
Evaluated independently for minimal and optimal levels.
Uses pydantic-ai Agent — model-agnostic. No Opik imports.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent

from mcp_eval.evaluators.core.prompts import trajectory_adherence as prompts
from mcp_eval.tasks.loader import ToolChain, ToolChainLevel

logger = logging.getLogger(__name__)


class _TrajectoryJudgment(BaseModel):
    score: float
    explanation: str

    @field_validator("score")
    @classmethod
    def clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


@dataclass
class TrajectoryOutput:
    score_minimal: float
    score_optimal: float
    explanation_minimal: str
    explanation_optimal: str


async def _judge_level(
    model: str,
    level: ToolChainLevel,
    actual_tool_calls: list[dict[str, Any]],
    user_prompt: str,
) -> _TrajectoryJudgment:
    agent: Agent[None, _TrajectoryJudgment] = Agent(
        model=model,
        system_prompt=prompts.SYSTEM_PROMPT,
        output_type=_TrajectoryJudgment,
    )
    required_tools_dicts = [{"name": t.name} for t in level.required_tools]
    user_msg = prompts.build_user_message(
        expected_chain=level.chain,
        required_tools=required_tools_dicts,
        actual_tool_calls=actual_tool_calls,
        user_prompt=user_prompt,
    )
    try:
        result = await agent.run(user_msg)
        return result.output
    except Exception as exc:
        logger.warning("Trajectory judge failed: %s", exc)
        return _TrajectoryJudgment(score=0.0, explanation=f"Judge error: {exc}")


async def compute_trajectory_adherence(
    model: str,
    tool_chain: ToolChain,
    actual_tool_calls: list[dict[str, Any]],
    user_prompt: str,
) -> TrajectoryOutput:
    """Two concurrent LLM calls: one for minimal chain, one for optimal chain."""
    minimal_result, optimal_result = await asyncio.gather(
        _judge_level(model, tool_chain.minimal, actual_tool_calls, user_prompt),
        _judge_level(model, tool_chain.optimal, actual_tool_calls, user_prompt),
    )
    return TrajectoryOutput(
        score_minimal=minimal_result.score,
        score_optimal=optimal_result.score,
        explanation_minimal=minimal_result.explanation,
        explanation_optimal=optimal_result.explanation,
    )
