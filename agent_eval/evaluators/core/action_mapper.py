"""
Action call mapper: classifies each actual tool call by capability category
and maps it to one or more semantic *action instances* (action name + arguments).

One literal tool call can yield 0..N semantic actions (e.g. several API searches
inside a single execute_python). Classification is deterministic where possible
(MCP tools without criteria, datagouv CLI, known web tools) and falls back to an
LLM judge for:
  - execute_python / execute_cli code
  - MCP tool calls with disambiguation criteria (e.g. query_resource_data)
  - unrecognised tool names

This module is the single engine feeding the action_* metrics (usage, params,
trajectory). The flat ordered `instances` list is the semantic source of truth.
"""

import asyncio
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

# Categories that, on a code/CLI execution tool, warrant an LLM pass to extract
# the embedded semantic action(s) and their arguments.
_CODE_CATEGORIES = (
    CATEGORY_DATAGOUV_API_HTTP,
    CATEGORY_DATAGOUV_CLI,
    CATEGORY_PYTHON_LOCAL,
    CATEGORY_FILE_LOAD_EXTERNAL,
)
_CODE_TOOLS = ("execute_python", "execute_cli")

_NON_ACTIONS = ("python_local_analysis", "unclassified")

# Module-level cache so actions.yml / action_args.yml are parsed only once per path.
_resolver_cache: dict[Path, SemanticLayerResolver] = {}


def _get_resolver(semantic_layer_dir: Path) -> SemanticLayerResolver:
    if semantic_layer_dir not in _resolver_cache:
        _resolver_cache[semantic_layer_dir] = SemanticLayerResolver(semantic_layer_dir)
    return _resolver_cache[semantic_layer_dir]


def _is_error(result: Any) -> bool:
    """Return True if a tool call result looks like an error."""
    if result is None:
        return False
    s = str(result).lower()
    return s.startswith("error") or "exception" in s or "traceback" in s


@dataclass
class ActionInstance:
    """One semantic action derived from a literal tool call."""

    action: str  # semantic action name, e.g. "search.datasets"
    args: dict  # arguments for this action instance
    source_tool_call_id: str
    source_tool_name: str
    capability_category: str
    confidence: float = 1.0
    errored: bool = False  # inherited from the source call's result
    reason: str = ""


@dataclass
class CallMapping:
    """How one literal tool call maps to semantic action instances."""

    tool_call_id: str
    tool_name: str
    capability_category: str
    errored: bool
    actions: list[ActionInstance] = field(default_factory=list)
    criteria_pending: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    reason: str = ""


@dataclass
class ActionMapperOutput:
    call_mappings: list[CallMapping] = field(default_factory=list)

    @property
    def instances(self) -> list[ActionInstance]:
        """Flat, ordered list of semantic action instances across all calls."""
        out: list[ActionInstance] = []
        for cm in self.call_mappings:
            out.extend(cm.actions)
        return out

    @property
    def counts(self) -> dict[str, int]:
        """Per-capability-category counts over literal tool calls."""
        result: dict[str, int] = {}
        for cm in self.call_mappings:
            result[cm.capability_category] = result.get(cm.capability_category, 0) + 1
        return result

    @property
    def mapped_fraction(self) -> float:
        """Fraction of literal tool calls mapped to >= 1 semantic action."""
        if not self.call_mappings:
            return 0.0
        mapped = sum(1 for cm in self.call_mappings if cm.actions)
        return mapped / len(self.call_mappings)

    @property
    def action_success_rate(self) -> float:
        """Fraction of semantic action instances whose source call did not error."""
        inst = self.instances
        if not inst:
            return 0.0
        ok = sum(1 for i in inst if not i.errored)
        return round(ok / len(inst), 6)


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
    for action in resolver.action_names():
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
    for action in resolver.action_names():
        for entry in resolver.resolve_tool_entries(action, "cli", ""):
            if entry.name and stripped.startswith(entry.name):
                matches.append(action)
    return matches


def _classify_deterministic(
    call: dict[str, Any],
    available_tool_names: list[str],
    mcp_version: str,
    resolver: SemanticLayerResolver,
) -> CallMapping:
    """
    Deterministically classify a tool call into a CallMapping.

    MCP tools without criteria yield confirmed action instances immediately.
    MCP tools with criteria entries return criteria_pending for the LLM judge.
    Code/CLI executions are categorised but left for the judge to extract actions.
    """
    name = call.get("name", "")
    call_id = call.get("tool_call_id", "")
    args = call.get("arguments") or {}
    errored = _is_error(call.get("result"))

    def cm(
        category: str, *, actions=None, criteria_pending=None, confidence=1.0, reason=""
    ) -> CallMapping:
        return CallMapping(
            tool_call_id=call_id,
            tool_name=name,
            capability_category=category,
            errored=errored,
            actions=actions or [],
            criteria_pending=criteria_pending or [],
            confidence=confidence,
            reason=reason,
        )

    def instances(
        action_names, category, confidence, reason, inst_args=None
    ) -> list[ActionInstance]:
        resolved_args = (
            inst_args
            if inst_args is not None
            else (args if isinstance(args, dict) else {})
        )
        return [
            ActionInstance(
                action=a,
                args=dict(resolved_args),
                source_tool_call_id=call_id,
                source_tool_name=name,
                capability_category=category,
                confidence=confidence,
                errored=errored,
                reason=reason,
            )
            for a in action_names
        ]

    # ── MCP tool ──────────────────────────────────────────────────────────────
    if name in available_tool_names:
        candidates = _mcp_name_to_semantic_candidates(name, mcp_version, resolver)
        confirmed = [a for a, c in candidates if c is None]
        pending = [(a, c) for a, c in candidates if c is not None]
        return cm(
            CATEGORY_MCP_TOOL,
            actions=instances(
                confirmed,
                CATEGORY_MCP_TOOL,
                1.0,
                "MCP tool name in available_tool_names",
            ),
            criteria_pending=pending,
            confidence=1.0,
            reason="MCP tool name in available_tool_names",
        )

    # ── Known web tools ────────────────────────────────────────────────────────
    if name == "duckduckgo_search":
        return cm(CATEGORY_WEB_SEARCH, confidence=1.0, reason="DuckDuckGo search tool")
    if name == "http_fetch":
        return cm(
            CATEGORY_WEB_PAGE_FETCH,
            confidence=1.0,
            reason="HTTP page fetch tool (web_search capability)",
        )

    # ── execute_cli ──────────────────────────────────────────────────────────────
    if name == "execute_cli":
        command = args.get("command", "") if isinstance(args, dict) else str(args)
        if command.strip().startswith("datagouv "):
            semantic = _datagouv_cli_to_semantic(command, resolver)
            return cm(
                CATEGORY_DATAGOUV_CLI,
                actions=instances(
                    semantic,
                    CATEGORY_DATAGOUV_CLI,
                    0.9,
                    f"datagouv CLI: {command[:80]}",
                    inst_args={"command": command},
                ),
                confidence=0.9 if semantic else 0.7,
                reason=f"datagouv CLI: {command[:80]}"
                + (f" → {semantic}" if semantic else " (LLM will refine)"),
            )
        if _contains_datagouv_api_url(command):
            return cm(
                CATEGORY_DATAGOUV_API_HTTP,
                confidence=0.8,
                reason="CLI command contains data.gouv.fr API URL",
            )
        return cm(
            CATEGORY_UNCLASSIFIED,
            confidence=0.5,
            reason="CLI command — not datagouv prefix",
        )

    # ── execute_python: quick heuristics ─────────────────────────────────────────
    if name == "execute_python":
        code = args.get("code", "") if isinstance(args, dict) else str(args)
        if _contains_datagouv_api_url(code):
            return cm(
                CATEGORY_DATAGOUV_API_HTTP,
                confidence=0.8,
                reason="Python code contains data.gouv.fr API URL — LLM will refine",
            )
        if not _contains_external_url(code):
            return cm(
                CATEGORY_PYTHON_LOCAL,
                confidence=0.7,
                reason="Python code with no external URLs — local analysis",
            )
        return cm(
            CATEGORY_FILE_LOAD_EXTERNAL,
            confidence=0.7,
            reason="Python code fetches external (non-datagouv) URL",
        )

    # ── Unknown tool — needs LLM ─────────────────────────────────────────────────
    return cm(CATEGORY_UNCLASSIFIED, confidence=0.0, reason="Tool name not recognised")


# ── LLM judge ─────────────────────────────────────────────────────────────────


class _JudgeActionEntry(BaseModel):
    action: str
    args: dict = {}
    confidence: float = 0.0
    reason: str = ""


class _JudgeResponse(BaseModel):
    actions: list[_JudgeActionEntry]


async def _judge_call(
    model: JudgeModel,
    call: dict[str, Any],
    semantic_actions: list[str],
    criteria_pending: list[tuple[str, str]] | None = None,
) -> tuple[list[tuple[str, dict]], float, str]:
    """
    Use the LLM judge to determine semantic action instance(s) for a tool call.

    Returns (confirmed_actions, max_confidence, reason) where confirmed_actions is
    a list of (action_name, extracted_args). For MCP disambiguation, criteria_pending
    is supplied and the judge decides which criteria-gated actions apply.
    """
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    args = call.get("arguments") or {}
    code = (
        args.get("code", args.get("command", ""))
        if isinstance(args, dict)
        else str(args)
    )
    output = str(call.get("result") or "")[:500]
    actions_list = "\n".join(f"- {a}" for a in semantic_actions)

    criteria_section = ""
    if criteria_pending:
        lines = ["\n## Disambiguation criteria\n"]
        lines += [f"- `{a}` applies if: {c}" for a, c in criteria_pending]
        criteria_section = "\n".join(lines)

    prompt = (
        prompt_template.format(
            semantic_actions_list=actions_list,
            tool_name=call.get("name", ""),
            code=code[:1000],
            output=output,
        )
        + criteria_section
    )

    agent = Agent(model=model, output_type=_JudgeResponse)
    try:
        result = await agent.run(prompt)
        entries = result.output.actions
        if not entries:
            return [], 0.0, "Judge returned no actions"
        qualified = [e for e in entries if e.confidence >= 0.5]
        if not qualified:
            return [], 0.0, "No actions above confidence threshold"
        max_confidence = max(e.confidence for e in qualified)
        reason = "; ".join(e.reason for e in qualified[:2])
        confirmed = [
            (e.action, e.args or {}) for e in qualified if e.action not in _NON_ACTIONS
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
    resolver = _get_resolver(semantic_layer_dir)
    all_semantic_actions = resolver.action_names()

    output = ActionMapperOutput()

    # ── Pass 1: deterministic classification ─────────────────────────────────────
    for call in actual_tool_calls:
        output.call_mappings.append(
            _classify_deterministic(call, available_tool_names, mcp_version, resolver)
        )

    if not judge_model:
        return output

    # ── Pass 2: LLM judge for criteria-pending + unclassified + code/CLI tools ────
    pending_idx: list[int] = []
    for i, cm in enumerate(output.call_mappings):
        needs_judge = (
            cm.capability_category == CATEGORY_UNCLASSIFIED
            or len(cm.criteria_pending) > 0
            or (
                cm.tool_name in _CODE_TOOLS
                and not cm.actions
                and cm.capability_category in _CODE_CATEGORIES
            )
        )
        if needs_judge:
            pending_idx.append(i)

    async def _judge(idx: int) -> tuple[int, list[tuple[str, dict]], float, str]:
        cm = output.call_mappings[idx]
        confirmed, confidence, reason = await _judge_call(
            judge_model,
            actual_tool_calls[idx],
            all_semantic_actions,
            criteria_pending=cm.criteria_pending or None,
        )
        return idx, confirmed, confidence, reason

    results = await asyncio.gather(*[_judge(i) for i in pending_idx])
    for idx, confirmed, confidence, reason in results:
        cm = output.call_mappings[idx]
        if not confirmed:
            continue
        is_mcp = cm.capability_category == CATEGORY_MCP_TOOL
        call_args = actual_tool_calls[idx].get("arguments") or {}
        existing = {inst.action for inst in cm.actions}
        for action, judge_args in confirmed:
            if action in existing:
                continue
            # MCP args are authoritative & structured; code/CLI args come from the judge.
            inst_args = (
                call_args
                if (is_mcp and isinstance(call_args, dict))
                else (judge_args or {})
            )
            cm.actions.append(
                ActionInstance(
                    action=action,
                    args=inst_args,
                    source_tool_call_id=cm.tool_call_id,
                    source_tool_name=cm.tool_name,
                    capability_category=cm.capability_category,
                    confidence=confidence,
                    errored=cm.errored,
                    reason=reason,
                )
            )
            existing.add(action)
        cm.confidence = max(cm.confidence, confidence)
        cm.reason = reason
        cm.criteria_pending = []

    return output
