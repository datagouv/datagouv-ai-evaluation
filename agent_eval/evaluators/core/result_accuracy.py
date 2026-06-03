"""
LLM-as-a-judge for result accuracy (criteria validation).
Uses pydantic-ai Agent so any supported provider/model can be used.
No Opik imports.
"""

import asyncio
import logging
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent

from agent_eval.evaluators.core._math import safe_div
from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.evaluators.core.prompts import result_accuracy as prompts
from agent_eval.tasks.loader import EvaluationCriteria

logger = logging.getLogger(__name__)


class _CriterionJudgment(BaseModel):
    validated: bool
    explanation: str


@dataclass
class CriterionResult:
    criterion: str
    validated: bool
    explanation: str


@dataclass
class ResultAccuracyOutput:
    results_minimal: list[CriterionResult]
    results_optimal: list[CriterionResult]
    total_criteria_minimal: int
    total_criteria_optimal: int
    total_validated_minimal: int
    total_validated_optimal: int
    result_accuracy_minimal: float
    result_accuracy_optimal: float


async def judge_criterion(
    model: JudgeModel,
    user_prompt: str,
    agent_answer: str,
    criterion: str,
) -> CriterionResult:
    """Single pydantic-ai Agent call to judge one criterion."""
    agent: Agent[None, _CriterionJudgment] = Agent(
        model=model,
        system_prompt=prompts.SYSTEM_PROMPT,
        output_type=_CriterionJudgment,
    )
    user_msg = prompts.build_user_message(user_prompt, agent_answer, criterion)
    try:
        result = await agent.run(user_msg)
        judgment = result.output
        return CriterionResult(
            criterion=criterion,
            validated=judgment.validated,
            explanation=judgment.explanation,
        )
    except Exception as exc:
        logger.warning("Criterion judge failed for %r: %s", criterion[:60], exc)
        return CriterionResult(
            criterion=criterion, validated=False, explanation=f"Judge error: {exc}"
        )


async def compute_result_accuracy(
    model: JudgeModel,
    user_prompt: str,
    agent_answer: str,
    evaluation_criteria: EvaluationCriteria,
) -> ResultAccuracyOutput:
    """
    Judge all criteria concurrently.
    Since optimal = minimal + extra, we deduplicate: judge each unique criterion once,
    then split results back into minimal / optimal buckets.
    """
    # Collect unique criteria preserving order
    seen: set[str] = set()
    unique_criteria: list[str] = []
    for c in evaluation_criteria.minimal + evaluation_criteria.optimal:
        if c not in seen:
            seen.add(c)
            unique_criteria.append(c)

    raw_results = await asyncio.gather(
        *[judge_criterion(model, user_prompt, agent_answer, c) for c in unique_criteria]
    )
    result_map: dict[str, CriterionResult] = {r.criterion: r for r in raw_results}

    results_minimal = [result_map[c] for c in evaluation_criteria.minimal]
    results_optimal = [result_map[c] for c in evaluation_criteria.optimal]

    total_min = len(results_minimal)
    total_opt = len(results_optimal)
    validated_min = sum(1 for r in results_minimal if r.validated)
    validated_opt = sum(1 for r in results_optimal if r.validated)

    return ResultAccuracyOutput(
        results_minimal=results_minimal,
        results_optimal=results_optimal,
        total_criteria_minimal=total_min,
        total_criteria_optimal=total_opt,
        total_validated_minimal=validated_min,
        total_validated_optimal=validated_opt,
        result_accuracy_minimal=safe_div(validated_min, total_min),
        result_accuracy_optimal=safe_div(validated_opt, total_opt),
    )
