"""
LLM-as-a-judge for trajectory adherence (sequence alignment).
Evaluated independently for minimal and optimal levels.
Operates on the mapped semantic action sequence (from action_mapper), not literal
tool calls. Uses pydantic-ai Agent — model-agnostic. No Opik imports.
"""

import asyncio
import logging
from dataclasses import dataclass

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent

from agent_eval.evaluators.core.action_mapper import ActionInstance
from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.evaluators.core.prompts import trajectory_adherence as prompts
from agent_eval.tasks.loader import ActionChain, ActionChainLevel

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
    model: JudgeModel,
    level: ActionChainLevel,
    instances: list[ActionInstance],
    user_prompt: str,
) -> _TrajectoryJudgment:
    agent: Agent[None, _TrajectoryJudgment] = Agent(
        model=model,
        system_prompt=prompts.SYSTEM_PROMPT,
        output_type=_TrajectoryJudgment,
    )
    required_actions_dicts = [{"name": t.name} for t in level.required_actions]
    actual_actions = [{"name": i.action, "arguments": i.args} for i in instances]
    user_msg = prompts.build_user_message(
        expected_chain=level.chain,
        required_actions=required_actions_dicts,
        actual_actions=actual_actions,
        user_prompt=user_prompt,
    )
    try:
        result = await agent.run(user_msg)
        return result.output
    except Exception as exc:
        logger.warning("Trajectory judge failed: %s", exc)
        return _TrajectoryJudgment(score=0.0, explanation=f"Judge error: {exc}")


async def compute_trajectory_adherence(
    model: JudgeModel,
    action_chain: ActionChain,
    instances: list[ActionInstance],
    user_prompt: str,
) -> TrajectoryOutput:
    """Two concurrent LLM calls: one for minimal chain, one for optimal chain."""
    minimal_result, optimal_result = await asyncio.gather(
        _judge_level(model, action_chain.minimal, instances, user_prompt),
        _judge_level(model, action_chain.optimal, instances, user_prompt),
    )
    return TrajectoryOutput(
        score_minimal=minimal_result.score,
        score_optimal=optimal_result.score,
        explanation_minimal=minimal_result.explanation,
        explanation_optimal=optimal_result.explanation,
    )
