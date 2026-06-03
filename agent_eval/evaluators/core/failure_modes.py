"""
LLM-as-a-judge for failure mode detection.
Uses pydantic-ai Agent — model-agnostic. No Opik imports.
Silently skips if failure_modes.yml is empty or not yet populated.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from agent_eval.evaluators.core.result_accuracy import CriterionResult
from agent_eval.evaluators.core.prompts import failure_modes as prompts
from agent_eval.evaluators.core.judge_model import JudgeModel

logger = logging.getLogger(__name__)

_DEFAULT_FAILURE_MODES_PATH = Path(__file__).parent / "config" / "failure_modes.yml"


class _FailureModeScores(BaseModel):
    present: list[str] = []
    explanations: dict[str, str] = {}


@dataclass
class FailureModeOutput:
    scores: dict[str, int] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)


async def judge_failure_modes(
    model: JudgeModel,
    user_prompt: str,
    agent_answer: str,
    failed_criteria: list[CriterionResult] | None = None,
    actual_tool_calls: list[dict[str, Any]] | None = None,
    required_actions_minimal: list[dict[str, Any]] | None = None,
    required_actions_optimal: list[dict[str, Any]] | None = None,
    failure_modes_path: Path | None = None,
) -> FailureModeOutput:
    """
    Run failure mode detection for one task.
    scores: {mode_name: 0|1} — 1 means the failure mode is present (higher = worse).
    Returns empty FailureModeOutput if no failure modes are defined yet.
    """
    failure_modes = prompts.load_failure_modes(
        failure_modes_path or _DEFAULT_FAILURE_MODES_PATH
    )
    if not failure_modes:
        return FailureModeOutput()

    agent: Agent[None, _FailureModeScores] = Agent(
        model=model,
        system_prompt=prompts.SYSTEM_PROMPT,
        output_type=_FailureModeScores,
    )
    user_msg = prompts.build_user_message(
        user_prompt=user_prompt,
        answer=agent_answer,
        failed_criteria=failed_criteria or [],
        actual_tool_calls=actual_tool_calls or [],
        required_tools_minimal=required_actions_minimal or [],
        required_tools_optimal=required_actions_optimal or [],
        failure_modes=failure_modes,
    )

    try:
        result = await agent.run(user_msg)
        present = set(result.output.present)
        explanations = result.output.explanations
        return FailureModeOutput(
            scores={
                fm["name"]: 1 if fm["name"] in present else 0 for fm in failure_modes
            },
            explanations={
                fm["name"]: explanations.get(fm["name"], "") for fm in failure_modes
            },
        )
    except Exception as exc:
        logger.warning("Failure mode judge failed: %s", exc)
        return FailureModeOutput(scores={fm["name"]: 0 for fm in failure_modes})
