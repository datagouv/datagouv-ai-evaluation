from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class RequiredToolArg:
    name: str
    strict_value: Any = None
    criteria: str | None = None


@dataclass
class RequiredTool:
    name: str
    args: list[RequiredToolArg] = field(default_factory=list)


@dataclass
class ToolChainLevel:
    chain: str
    required_tools: list[RequiredTool] = field(default_factory=list)


@dataclass
class ToolChain:
    minimal: ToolChainLevel
    optimal: ToolChainLevel  # always minimal + optimal combined


@dataclass
class EvaluationCriteria:
    minimal: list[str]
    optimal: list[str]  # always minimal + optimal combined


@dataclass
class ResourceCheck:
    fn: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    type: str
    id: str
    dataset_id: Optional[str]
    checks: list[ResourceCheck] = field(default_factory=list)


@dataclass
class TaskMeta:
    status: str
    source: str
    turn: str
    minimal_tool_invocation_type: str
    optimal_tool_invocation_type: str


@dataclass
class Task:
    task_id: str
    task_name: str
    v_introduced: str
    meta: TaskMeta
    prompt: str
    evaluation_criteria: EvaluationCriteria
    tool_chain: ToolChain
    resources: list[Resource] = field(default_factory=list)


# ── Version resolution ────────────────────────────────────────────────────────


def _version_key(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


def _resolve_latest(entries: list[dict]) -> dict:
    """Return the entry with the highest v_introduced."""
    return max(entries, key=lambda e: _version_key(e.get("v_introduced", "0")))


# ── Parsers ───────────────────────────────────────────────────────────────────


def _parse_required_tool_arg(raw: dict) -> RequiredToolArg:
    return RequiredToolArg(
        name=raw.get("name") or "",
        strict_value=raw.get("strict_value"),
        criteria=raw.get("criteria"),
    )


def _parse_required_tool(raw: dict) -> RequiredTool:
    args = [_parse_required_tool_arg(a) for a in (raw.get("args") or [])]
    return RequiredTool(name=raw["name"], args=args)


def _parse_tool_chain_level(
    raw: dict | None, extra_tools: list[RequiredTool] | None = None
) -> ToolChainLevel:
    if raw is None:
        return ToolChainLevel(chain="", required_tools=list(extra_tools or []))
    tools = [_parse_required_tool(t) for t in (raw.get("required_tools") or [])]
    if extra_tools:
        tools = list(extra_tools) + tools
    return ToolChainLevel(
        chain=raw.get("chain") or "",
        required_tools=tools,
    )


def _build_tool_chain(entries: list[dict]) -> ToolChain:
    entry = _resolve_latest(entries)
    minimal_level = _parse_tool_chain_level(entry.get("minimal"))
    raw_optimal = entry.get("optimal")
    if raw_optimal and (raw_optimal.get("required_tools") or raw_optimal.get("chain")):
        # optimal = minimal tools + optimal additional tools
        optimal_level = _parse_tool_chain_level(
            raw_optimal, extra_tools=minimal_level.required_tools
        )
    else:
        # no optimal block → optimal equals minimal
        optimal_level = ToolChainLevel(
            chain=minimal_level.chain,
            required_tools=list(minimal_level.required_tools),
        )
    return ToolChain(minimal=minimal_level, optimal=optimal_level)


def _build_evaluation_criteria(entries: list[dict]) -> EvaluationCriteria:
    entry = _resolve_latest(entries)
    minimal = [c["criteria"] for c in (entry.get("minimal") or []) if c.get("criteria")]
    optimal_extra = [
        c["criteria"] for c in (entry.get("optimal") or []) if c.get("criteria")
    ]
    if optimal_extra:
        optimal = minimal + optimal_extra
    else:
        # no optimal criteria → optimal equals minimal
        optimal = list(minimal)
    return EvaluationCriteria(minimal=minimal, optimal=optimal)


def _parse_resource_check(raw: dict) -> ResourceCheck:
    return ResourceCheck(fn=raw["fn"], params=raw.get("params") or {})


def _parse_resource(raw: dict) -> Resource:
    checks = [_parse_resource_check(c) for c in (raw.get("checks") or [])]
    return Resource(type=raw["type"], id=str(raw["id"]), dataset_id=raw.get("dataset_id"), checks=checks)


def _parse_task_meta(raw: dict) -> TaskMeta:
    return TaskMeta(
        status=raw.get("status", "draft"),
        source=raw.get("source", ""),
        turn=raw.get("turn", "single"),
        minimal_tool_invocation_type=raw.get("minimal_tool_invocation_type", ""),
        optimal_tool_invocation_type=raw.get("optimal_tool_invocation_type", ""),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def load_task(path: Path) -> Task:
    """Load and parse a single task YAML file into a Task object."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ec_entries = raw.get("evaluation_criteria") or []
    best_ec = _resolve_latest(ec_entries) if ec_entries else {}
    v_introduced = str(best_ec.get("v_introduced", "1.0"))

    return Task(
        task_id=str(raw["task_id"]),
        task_name=str(raw.get("task_name", "")),
        v_introduced=v_introduced,
        meta=_parse_task_meta(raw.get("meta") or {}),
        prompt=str(raw.get("prompt") or "").strip(),
        evaluation_criteria=_build_evaluation_criteria(ec_entries),
        tool_chain=_build_tool_chain(raw.get("tool_chain") or []),
        resources=[_parse_resource(r) for r in (raw.get("resources") or [])],
    )


def load_all_tasks(tasks_dir: Path, status_filter: str = "active") -> list[Task]:
    """
    Load all task_*.yml files from tasks_dir, filtered by meta.status.
    Skips example_task_0000.yml and any file not matching task_*.yml.
    """
    paths = sorted(tasks_dir.glob("task_*.yml"))
    tasks = []
    for path in paths:
        task = load_task(path)
        if task.meta.status == status_filter:
            tasks.append(task)
    return tasks


def task_to_opik_item(task: Task) -> dict:
    """Convert a Task into an Opik DatasetItem-compatible dict."""

    def tool_chain_level_to_dict(level: ToolChainLevel) -> dict:
        return {
            "chain": level.chain,
            "required_tools": [
                {
                    "name": t.name,
                    "args": [
                        {
                            "name": a.name,
                            "strict_value": a.strict_value,
                            "criteria": a.criteria,
                        }
                        for a in t.args
                    ],
                }
                for t in level.required_tools
            ],
        }

    return {
        "input": {
            "prompt": task.prompt,
            "task_id": task.task_id,
        },
        "expected_output": {
            "evaluation_criteria": {
                "minimal": task.evaluation_criteria.minimal,
                "optimal": task.evaluation_criteria.optimal,
            },
            "tool_chain": {
                "minimal": tool_chain_level_to_dict(task.tool_chain.minimal),
                "optimal": tool_chain_level_to_dict(task.tool_chain.optimal),
            },
        },
        "metadata": {
            "task_name": task.task_name,
            "source": task.meta.source,
            "turn": task.meta.turn,
            "minimal_tool_invocation_type": task.meta.minimal_tool_invocation_type,
            "optimal_tool_invocation_type": task.meta.optimal_tool_invocation_type,
        },
    }
