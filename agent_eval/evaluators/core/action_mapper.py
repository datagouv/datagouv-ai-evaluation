"""
Action call mapper: classifies each actual_tool_call by capability category
and maps it to one or more semantic actions.

Classification is deterministic where possible (MCP tools without criteria,
datagouv CLI, known web tools) and falls back to an LLM judge for:
  - execute_python calls
  - MCP tool calls with disambiguation criteria (e.g. query_resource_data)
  - unrecognised tool names
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.semantic_layer import SemanticLayerResolver

_PROMPT_PATH = Path(__file__).parent / "prompts" / "action_mapper.txt"

# URL patterns that identify data.gouv.fr API calls
_DATAGOUV_API_PATTERNS = (
    "data.gouv.fr/api/",
    "tabular-api.data.gouv.fr/",
)

# Capability categories
CATEGORY_MCP_TOOL = "mcp_tool"
CATEGORY_DATAGOUV_CLI = "datagouv_cli"
CATEGORY_DATAGOUV_API_HTTP = "datagouv_api_http"
CATEGORY_PYTHON_LOCAL = "python_local_analysis"
CATEGORY_WEB_SEARCH = "web_search"
CATEGORY_WEB_PAGE_FETCH = "web_page_fetch"
CATEGORY_FILE_LOAD_EXTERNAL = "file_load_external"
CATEGORY_UNCLASSIFIED = "unclassified"

_ALL_SEMANTIC_ACTIONS = [
    "search.datasets", "search.dataservices",
    "get.dataset.info", "get.dataset.resources",
    "get.resource.info", "get.resource.profile",
    "get.data",
    "get.dataservice.info", "get.dataservice.openapi_spec",
]


@dataclass
class ActionMapping:
    tool_call_id: str
    tool_name: str
    capability_category: str
    semantic_actions: list[str]                  # confirmed (criteria=None or LLM-confirmed)
    criteria_pending: list[tuple[str, str]]      # [(action, criteria)] awaiting LLM check
    confidence: float
    reason: str = ""


@dataclass
class ActionMapperOutput:
    mappings: list[ActionMapping] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for m in self.mappings:
            result[m.capability_category] = result.get(m.capability_category, 0) + 1
        return result

    @property
    def mapped_fraction(self) -> float:
        if not self.mappings:
            return 0.0
        mapped = sum(1 for m in self.mappings if len(m.semantic_actions) > 0)
        return mapped / len(self.mappings)


def _contains_datagouv_api_url(text: str) -> bool:
    return any(p in text for p in _DATAGOUV_API_PATTERNS)


def _contains_external_url(text: str) -> bool:
    return bool(re.search(r"https?://", text))


def _mcp_name_to_semantic_candidates(
    tool_name: str,
    mcp_version: str,
    resolver: SemanticLayerResolver,
) -> list[tuple[str, str | None]]:
    """
    Return [(semantic_action, criteria_or_None)] for all semantic actions
    whose MCP entries for this version include the given tool name.
    Most tools return one entry; query_resource_data returns two
    (get.resource.profile with criteria, get.data without).
    """
    results = []
    for action in _ALL_SEMANTIC_ACTIONS:
        for entry in resolver.resolve_tool_entries(action, "mcp", mcp_version):
            if entry.name == tool_name:
                results.append((action, entry.criteria))
    return results


def _datagouv_cli_to_semantic(
    command: str,
    resolver: SemanticLayerResolver,
) -> list[str]:
    """Prefix-match a datagouv CLI command against CLI command patterns in the semantic layer."""
    stripped = command.strip()
    matches = []
    for action in _ALL_SEMANTIC_ACTIONS:
        for entry in resolver.resolve_tool_entries(action, "cli", ""):
            if entry.name and stripped.startswith(entry.name):
                matches.append(action)
    return matches


def _classify_deterministic(
    call: dict[str, Any],
    available_tool_names: list[str],
    mcp_version: str,
    resolver: SemanticLayerResolver,
) -> ActionMapping | None:
    """
    Deterministically classify a tool call.
    Returns None only for completely unrecognised tool names.

    MCP tools with criteria entries return a mapping with criteria_pending populated;
    the LLM judge must confirm which criteria-gated actions apply.
    """
    name = call.get("name", "")
    call_id = call.get("tool_call_id", "")

    # ── MCP tool ────────────────────────────────────────────────────────────────
    if name in available_tool_names:
        candidates = _mcp_name_to_semantic_candidates(name, mcp_version, resolver)
        confirmed = [a for a, c in candidates if c is None]
        pending = [(a, c) for a, c in candidates if c is not None]
        return ActionMapping(
            tool_call_id=call_id,
            tool_name=name,
            capability_category=CATEGORY_MCP_TOOL,
            semantic_actions=confirmed,
            criteria_pending=pending,
            confidence=1.0,
            reason="MCP tool name in available_tool_names",
        )

    # ── Known web tools ──────────────────────────────────────────────────────────
    if name == "duckduckgo_search":
        return ActionMapping(
            tool_call_id=call_id, tool_name=name,
            capability_category=CATEGORY_WEB_SEARCH,
            semantic_actions=[], criteria_pending=[], confidence=1.0,
            reason="DuckDuckGo search tool",
        )
    if name == "http_fetch":
        return ActionMapping(
            tool_call_id=call_id, tool_name=name,
            capability_category=CATEGORY_WEB_PAGE_FETCH,
            semantic_actions=[], criteria_pending=[], confidence=1.0,
            reason="HTTP page fetch tool (web_search capability)",
        )

    # ── execute_cli ──────────────────────────────────────────────────────────────
    if name == "execute_cli":
        args = call.get("arguments") or {}
        command = args.get("command", "") if isinstance(args, dict) else str(args)
        if command.strip().startswith("datagouv "):
            semantic = _datagouv_cli_to_semantic(command, resolver)
            return ActionMapping(
                tool_call_id=call_id, tool_name=name,
                capability_category=CATEGORY_DATAGOUV_CLI,
                semantic_actions=semantic, criteria_pending=[],
                confidence=0.9 if semantic else 0.7,
                reason=f"datagouv CLI: {command[:80]}" + (f" → {semantic}" if semantic else " (no semantic match)"),
            )
        if _contains_datagouv_api_url(command):
            return ActionMapping(
                tool_call_id=call_id, tool_name=name,
                capability_category=CATEGORY_DATAGOUV_API_HTTP,
                semantic_actions=[], criteria_pending=[], confidence=0.8,
                reason="CLI command contains data.gouv.fr API URL",
            )
        # Other CLI — no semantic mapping without LLM
        return ActionMapping(
            tool_call_id=call_id, tool_name=name,
            capability_category=CATEGORY_DATAGOUV_API_HTTP if _contains_datagouv_api_url(command) else CATEGORY_UNCLASSIFIED,
            semantic_actions=[], criteria_pending=[], confidence=0.5,
            reason="CLI command — not datagouv prefix",
        )

    # ── execute_python: quick heuristics ─────────────────────────────────────────
    if name == "execute_python":
        args = call.get("arguments") or {}
        code = args.get("code", "") if isinstance(args, dict) else str(args)
        if _contains_datagouv_api_url(code):
            return ActionMapping(
                tool_call_id=call_id, tool_name=name,
                capability_category=CATEGORY_DATAGOUV_API_HTTP,
                semantic_actions=[], criteria_pending=[], confidence=0.8,
                reason="Python code contains data.gouv.fr API URL — LLM will refine",
            )
        if not _contains_external_url(code):
            return ActionMapping(
                tool_call_id=call_id, tool_name=name,
                capability_category=CATEGORY_PYTHON_LOCAL,
                semantic_actions=[], criteria_pending=[], confidence=0.7,
                reason="Python code with no external URLs — local analysis",
            )
        return ActionMapping(
            tool_call_id=call_id, tool_name=name,
            capability_category=CATEGORY_FILE_LOAD_EXTERNAL,
            semantic_actions=[], criteria_pending=[], confidence=0.7,
            reason="Python code fetches external (non-datagouv) URL",
        )

    return None  # unknown tool — needs LLM


# ── LLM judge ─────────────────────────────────────────────────────────────────


class _JudgeResponse(BaseModel):
    actions: list[dict]


async def _judge_call(
    model: JudgeModel,
    call: dict[str, Any],
    semantic_actions: list[str],
    criteria_pending: list[tuple[str, str]] | None = None,
) -> tuple[list[str], float, str]:
    """
    Use the LLM judge to determine semantic action(s) for a tool call.

    If criteria_pending is provided, the prompt asks the judge to evaluate
    whether the actual tool call arguments satisfy each criteria string.
    Returns (confirmed_actions, max_confidence, reason).
    """
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    args = call.get("arguments") or {}
    code = args.get("code", args.get("command", "")) if isinstance(args, dict) else str(args)
    output = str(call.get("result") or "")[:500]
    actions_list = "\n".join(f"- {a}" for a in semantic_actions)

    # Append criteria hints when disambiguating MCP tools
    criteria_section = ""
    if criteria_pending:
        lines = ["\n## Disambiguation criteria\n"]
        lines += [f"- `{a}` applies if: {c}" for a, c in criteria_pending]
        criteria_section = "\n".join(lines)

    prompt = prompt_template.format(
        semantic_actions_list=actions_list,
        tool_name=call.get("name", ""),
        code=code[:1000],
        output=output,
    ) + criteria_section

    agent = Agent(model=model, output_type=_JudgeResponse)
    try:
        result = await agent.run(prompt)
        action_entries = result.output.actions
        if not action_entries:
            return [], 0.0, "Judge returned no actions"
        qualified = [e for e in action_entries if e.get("confidence", 0) >= 0.5]
        if not qualified:
            return [], 0.0, "No actions above confidence threshold"
        max_confidence = max(e.get("confidence", 0) for e in qualified)
        reason = "; ".join(e.get("reason", "") for e in qualified[:2])
        confirmed = [
            e["action"] for e in qualified
            if e.get("action") not in ("python_local_analysis", "unclassified")
        ]
        return confirmed, max_confidence, reason
    except Exception as exc:
        return [], 0.0, f"Judge error: {exc}"


async def map_action_calls(
    actual_tool_calls: list[dict[str, Any]],
    available_tool_names: list[str],
    mcp_version: str,
    semantic_layer_dir: Path,
    judge_model: JudgeModel | None = None,
) -> ActionMapperOutput:
    resolver = SemanticLayerResolver(semantic_layer_dir)
    all_semantic_actions = list(resolver._actions.keys())

    output = ActionMapperOutput()

    # ── Pass 1: deterministic classification ─────────────────────────────────────
    for call in actual_tool_calls:
        mapping = _classify_deterministic(call, available_tool_names, mcp_version, resolver)
        if mapping is not None:
            output.mappings.append(mapping)
        else:
            output.mappings.append(ActionMapping(
                tool_call_id=call.get("tool_call_id", ""),
                tool_name=call.get("name", ""),
                capability_category=CATEGORY_UNCLASSIFIED,
                semantic_actions=[], criteria_pending=[],
                confidence=0.0, reason="Tool name not recognised",
            ))

    if not judge_model:
        return output

    # ── Pass 2: LLM judge for criteria-pending + unclassified + code tools ────────
    import asyncio

    llm_pending: list[int] = []
    for i, (mapping, call) in enumerate(zip(output.mappings, actual_tool_calls)):
        needs_judge = (
            mapping.capability_category == CATEGORY_UNCLASSIFIED
            or len(mapping.criteria_pending) > 0
            or (
                mapping.capability_category in (CATEGORY_DATAGOUV_API_HTTP, CATEGORY_PYTHON_LOCAL, CATEGORY_FILE_LOAD_EXTERNAL)
                and mapping.tool_name == "execute_python"
                and len(mapping.semantic_actions) == 0
            )
        )
        if needs_judge:
            llm_pending.append(i)

    async def _judge(idx: int) -> tuple[int, list[str], float, str]:
        call = actual_tool_calls[idx]
        mapping = output.mappings[idx]
        actions, confidence, reason = await _judge_call(
            judge_model, call, all_semantic_actions,
            criteria_pending=mapping.criteria_pending or None,
        )
        return idx, actions, confidence, reason

    results = await asyncio.gather(*[_judge(i) for i in llm_pending])
    for idx, actions, confidence, reason in results:
        if actions:
            output.mappings[idx].semantic_actions.extend(
                a for a in actions if a not in output.mappings[idx].semantic_actions
            )
            output.mappings[idx].confidence = max(output.mappings[idx].confidence, confidence)
            output.mappings[idx].reason = reason
            # Criteria confirmed — move out of pending
            output.mappings[idx].criteria_pending = []

    return output
