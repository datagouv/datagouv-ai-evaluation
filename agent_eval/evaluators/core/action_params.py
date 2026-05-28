"""
LLM-as-a-judge for semantic-action parameter correctness.

Operates in semantic space: the actual inputs are the mapped action instances
(action name + extracted args) produced by action_mapper, not literal tool calls.
Uses pydantic-ai Agent — model-agnostic. No Opik imports.

Design:
- All ground-truth actions for the same semantic name are grouped into ONE prompt with
  all actual action instances of that name, so the judge can do one-to-one mapping.
- Redundancy is intra-group: if two actual actions match the same GT action, only the
  closest one is valid; the other is marked redundant.
- Schema compliance is NOT checked: the judge cannot know the full argument surface of
  non-MCP tools (execute_python/execute_cli), so an action is "validated" iff its listed
  parameters are correct.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from agent_eval.evaluators.core.action_mapper import ActionInstance
from agent_eval.evaluators.core.prompts import action_parameter_correctness as prompts
from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.tasks.loader import RequiredAction

logger = logging.getLogger(__name__)


# ── Pydantic output model for the LLM judge ──────────────────────────────────


class _MappingEntry(BaseModel):
    ground_truth_index: int
    actual_call_index: int
    correct: bool
    explanation: str


class _ActionJudgment(BaseModel):
    mappings: list[_MappingEntry]
    unmatched_actual_indices: list[int] = []
    redundant_actual_indices: list[int] = []


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class ActionMatch:
    action: str
    gt_index: int               # index in the required-actions group
    actual_index: int           # index in the actual-instances group
    correct: bool               # LLM says listed params are correct
    explanation: str
    source_tool_call_id: str = ""

    @property
    def validated(self) -> bool:
        return self.correct


@dataclass
class ActionParamsOutput:
    matches: list[ActionMatch] = field(default_factory=list)
    unmatched_actual_indices: list[int] = field(default_factory=list)
    redundant_actual_indices: list[int] = field(default_factory=list)

    @property
    def matched_actions(self) -> int:
        """GT actions that were correctly matched (params correct)."""
        return sum(1 for m in self.matches if m.validated)


# ── Judge one action group ──────────────────────────────────────────────────


async def _judge_action_group(
    model: JudgeModel,
    action_name: str,
    gt_actions: list[RequiredAction],
    actual_instances: list[ActionInstance],
    user_prompt: str,
) -> ActionParamsOutput:
    gt_dicts = [
        {
            "args": [
                {"name": a.name, "strict_value": a.strict_value, "criteria": a.criteria}
                for a in t.args
            ]
        }
        for t in gt_actions
    ]
    actual_dicts = [{"arguments": inst.args} for inst in actual_instances]

    agent: Agent[None, _ActionJudgment] = Agent(
        model=model,
        system_prompt=prompts.SYSTEM_PROMPT,
        output_type=_ActionJudgment,
    )
    user_msg = prompts.build_user_message(
        action_name, gt_dicts, actual_dicts, user_prompt
    )

    try:
        result = await agent.run(user_msg)
        judgment = result.output
    except Exception as exc:
        logger.warning("Action params judge failed for %r: %s", action_name, exc)
        return ActionParamsOutput()

    matches: list[ActionMatch] = []
    for entry in judgment.mappings:
        gi, ai = entry.ground_truth_index, entry.actual_call_index
        if gi >= len(gt_actions) or ai >= len(actual_instances):
            continue
        matches.append(
            ActionMatch(
                action=action_name,
                gt_index=gi,
                actual_index=ai,
                correct=entry.correct,
                explanation=entry.explanation,
                source_tool_call_id=actual_instances[ai].source_tool_call_id,
            )
        )

    return ActionParamsOutput(
        matches=matches,
        unmatched_actual_indices=judgment.unmatched_actual_indices,
        redundant_actual_indices=judgment.redundant_actual_indices,
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def judge_action_params(
    model: JudgeModel,
    user_prompt: str,
    instances: list[ActionInstance],
    required_actions: list[RequiredAction],
) -> ActionParamsOutput:
    """
    Evaluate parameter correctness for all required semantic actions.
    Groups by semantic action name and runs one judge call per group concurrently.
    Returns merged ActionParamsOutput across all groups.
    """
    gt_by_name: dict[str, list[RequiredAction]] = defaultdict(list)
    for t in required_actions:
        gt_by_name[t.name].append(t)

    actual_by_name: dict[str, list[ActionInstance]] = defaultdict(list)
    for inst in instances:
        actual_by_name[inst.action].append(inst)

    coros = [
        _judge_action_group(
            model,
            action_name,
            gt_actions,
            actual_by_name.get(action_name, []),
            user_prompt,
        )
        for action_name, gt_actions in gt_by_name.items()
    ]

    group_results: list[ActionParamsOutput] = list(await asyncio.gather(*coros))

    merged = ActionParamsOutput()
    for g in group_results:
        merged.matches.extend(g.matches)
        merged.unmatched_actual_indices.extend(g.unmatched_actual_indices)
        merged.redundant_actual_indices.extend(g.redundant_actual_indices)
    return merged
