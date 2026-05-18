"""
LLM-as-a-judge for tool parameter correctness.
Uses pydantic-ai Agent — model-agnostic. No Opik imports.

Design:
- All ground-truth calls for the same tool name are grouped into ONE prompt with
  all actual calls for that tool, so the judge can do one-to-one mapping.
- Redundancy is intra-group: if two actual calls match the same GT call, only the
  closest one is valid; the other is marked redundant.
- After the LLM mapping, schema compliance is checked deterministically for any
  arg not covered by criteria/strict_value constraints.
  A call that fails schema compliance is NOT counted as correct even if its
  listed parameters matched.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from mcp_eval.evaluators.core.schema_compliance import check_schema_compliance
from mcp_eval.evaluators.core.prompts import tool_parameter_correctness as prompts
from mcp_eval.evaluators.core.judge_model import JudgeModel
from mcp_eval.tasks.loader import RequiredTool

logger = logging.getLogger(__name__)


# ── Pydantic output model for the LLM judge ──────────────────────────────────


class _MappingEntry(BaseModel):
    ground_truth_index: int
    actual_call_index: int
    correct: bool
    explanation: str


class _ToolJudgment(BaseModel):
    mappings: list[_MappingEntry]
    unmatched_actual_indices: list[int] = []
    redundant_actual_indices: list[int] = []


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class ToolCallMatch:
    tool_name: str
    gt_index: int
    actual_index: int
    correct: bool  # LLM says listed params are correct
    schema_compliant: bool  # deterministic check on unlisted args
    explanation: str

    @property
    def validated(self) -> bool:
        return self.correct and self.schema_compliant


@dataclass
class ToolParamsOutput:
    matches: list[ToolCallMatch] = field(default_factory=list)
    unmatched_actual_indices: list[int] = field(default_factory=list)
    redundant_actual_indices: list[int] = field(default_factory=list)

    @property
    def ground_truth_tool_calls(self) -> int:
        """GT calls that were correctly matched (params correct + schema compliant)."""
        return sum(1 for m in self.matches if m.validated)

    @property
    def correct_parameters_tool_calls(self) -> int:
        """Actual calls with all listed params correct AND schema compliant."""
        return sum(1 for m in self.matches if m.validated)


# ── Judge one tool group ──────────────────────────────────────────────────────


async def _judge_tool_group(
    model: JudgeModel,
    tool_name: str,
    gt_calls: list[RequiredTool],
    actual_calls: list[dict[str, Any]],
    available_tools_schema: list[dict],
    user_prompt: str,
) -> ToolParamsOutput:
    gt_dicts = [
        {
            "args": [
                {"name": a.name, "strict_value": a.strict_value, "criteria": a.criteria}
                for a in t.args
            ]
        }
        for t in gt_calls
    ]

    agent: Agent[None, _ToolJudgment] = Agent(
        model=model,
        system_prompt=prompts.SYSTEM_PROMPT,
        output_type=_ToolJudgment,
    )
    user_msg = prompts.build_user_message(
        tool_name, gt_dicts, actual_calls, user_prompt
    )

    try:
        result = await agent.run(user_msg)
        judgment = result.output
    except Exception as exc:
        logger.warning("Tool params judge failed for %r: %s", tool_name, exc)
        return ToolParamsOutput()

    matches: list[ToolCallMatch] = []
    for entry in judgment.mappings:
        gi, ai = entry.ground_truth_index, entry.actual_call_index
        if gi >= len(gt_calls) or ai >= len(actual_calls):
            continue
        schema_ok = check_schema_compliance(actual_calls[ai], available_tools_schema)
        matches.append(
            ToolCallMatch(
                tool_name=tool_name,
                gt_index=gi,
                actual_index=ai,
                correct=entry.correct,
                schema_compliant=schema_ok,
                explanation=entry.explanation,
            )
        )

    return ToolParamsOutput(
        matches=matches,
        unmatched_actual_indices=judgment.unmatched_actual_indices,
        redundant_actual_indices=judgment.redundant_actual_indices,
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def judge_tool_params(
    model: JudgeModel,
    user_prompt: str,
    actual_tool_calls: list[dict[str, Any]],
    required_tools: list[RequiredTool],
    available_tools_schema: list[dict],
) -> ToolParamsOutput:
    """
    Evaluate parameter correctness for all required tool calls.
    Groups calls by tool name and runs one judge call per group concurrently.
    Returns merged ToolParamsOutput across all groups.
    """
    gt_by_name: dict[str, list[RequiredTool]] = defaultdict(list)
    for t in required_tools:
        gt_by_name[t.name].append(t)

    actual_by_name: dict[str, list[dict]] = defaultdict(list)
    for call in actual_tool_calls:
        actual_by_name[call.get("name", "")].append(call)

    coros = [
        _judge_tool_group(
            model,
            tool_name,
            gt_calls,
            actual_by_name.get(tool_name, []),
            available_tools_schema,
            user_prompt,
        )
        for tool_name, gt_calls in gt_by_name.items()
    ]

    group_results: list[ToolParamsOutput] = list(await asyncio.gather(*coros))

    merged = ToolParamsOutput()
    for g in group_results:
        merged.matches.extend(g.matches)
        merged.unmatched_actual_indices.extend(g.unmatched_actual_indices)
        merged.redundant_actual_indices.extend(g.redundant_actual_indices)
    return merged
