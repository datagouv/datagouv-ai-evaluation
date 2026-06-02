from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Fixed epoch anchor for deterministic UUID v7 item IDs (2024-01-01T00:00:00Z in ms).
# Opik 2.x requires UUID v7; using a fixed timestamp keeps IDs stable across runs.
_EPOCH_MS = 1704067200000


def _task_id_to_uuid7(task_id: str, version: str = "") -> str:
    """Deterministic UUID v7 derived from task_id (and optional dataset version).

    Version nibble = 7, variant = 10 (RFC 4122), timestamp fixed at _EPOCH_MS.
    Random bits come from SHA-256 of `task_id[:version]` — unique and stable per
    (task_id, version). Bumping `version` yields entirely new ids so a refreshed
    dataset doesn't collide with orphaned `dataset_item_versions` rows from prior
    submissions (Opik's API delete does NOT cascade to ClickHouse).
    """
    key = f"{task_id}:{version}" if version else task_id
    h = hashlib.sha256(key.encode()).digest()
    b = bytearray(16)
    # Bytes 0-5: 48-bit timestamp
    ts = _EPOCH_MS
    b[0] = (ts >> 40) & 0xFF
    b[1] = (ts >> 32) & 0xFF
    b[2] = (ts >> 24) & 0xFF
    b[3] = (ts >> 16) & 0xFF
    b[4] = (ts >> 8) & 0xFF
    b[5] = ts & 0xFF
    # Byte 6: version nibble = 7, upper 4 random bits
    b[6] = 0x70 | (h[0] & 0x0F)
    # Byte 7: next 8 random bits
    b[7] = h[1]
    # Byte 8: variant bits = 10xxxxxx
    b[8] = 0x80 | (h[2] & 0x3F)
    # Bytes 9-15: remaining random bits
    b[9:16] = h[3:10]
    return str(uuid.UUID(bytes=bytes(b)))


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class RequiredActionArg:
    name: str
    strict_value: Any = None
    criteria: str | None = None


@dataclass
class RequiredAction:
    name: str
    args: list[RequiredActionArg] = field(default_factory=list)


@dataclass
class ActionChainLevel:
    chain: str
    required_actions: list[RequiredAction] = field(default_factory=list)


@dataclass
class ActionChain:
    minimal: ActionChainLevel
    optimal: ActionChainLevel  # always minimal + optimal combined


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
    dataset_id: str | None
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
    action_chain: ActionChain
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


def _parse_required_action_arg(raw: dict) -> RequiredActionArg:
    return RequiredActionArg(
        name=raw.get("name") or "",
        strict_value=raw.get("strict_value"),
        criteria=raw.get("criteria"),
    )


def _parse_required_action(raw: dict) -> RequiredAction:
    args = [_parse_required_action_arg(a) for a in (raw.get("args") or [])]
    return RequiredAction(name=raw["name"], args=args)


def _parse_action_chain_level(
    raw: dict | None, extra_actions: list[RequiredAction] | None = None
) -> ActionChainLevel:
    if raw is None:
        return ActionChainLevel(chain="", required_actions=list(extra_actions or []))
    actions = [_parse_required_action(t) for t in (raw.get("required_actions") or [])]
    if extra_actions:
        actions = list(extra_actions) + actions
    return ActionChainLevel(
        chain=raw.get("chain") or "",
        required_actions=actions,
    )


def _build_action_chain(raw: dict) -> ActionChain:
    """Build ActionChain from a plain action_chain dict (no v_introduced wrapper)."""
    minimal_level = _parse_action_chain_level(raw.get("minimal"))
    raw_optimal = raw.get("optimal")
    if raw_optimal and (raw_optimal.get("required_actions") or raw_optimal.get("chain")):
        # optimal = minimal actions + optimal additional actions
        optimal_level = _parse_action_chain_level(
            raw_optimal, extra_actions=minimal_level.required_actions
        )
    else:
        # no optimal block → optimal equals minimal
        optimal_level = ActionChainLevel(
            chain=minimal_level.chain,
            required_actions=list(minimal_level.required_actions),
        )
    return ActionChain(minimal=minimal_level, optimal=optimal_level)


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
        action_chain=_build_action_chain(raw.get("action_chain") or {}),
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


def task_to_opik_item(task: Task, version: str = "") -> dict:
    """Convert a Task into an Opik DatasetItem-compatible dict.

    `version` is folded into the item id so bumping it yields fresh ids (used to
    cleanly re-seed the dataset when tasks change; see `DATASET_VERSION` in
    run_experiments.py).
    """

    def action_chain_level_to_dict(level: ActionChainLevel) -> dict:
        return {
            "chain": level.chain,
            "required_actions": [
                {
                    "name": a.name,
                    "args": [
                        {
                            "name": arg.name,
                            "strict_value": arg.strict_value,
                            "criteria": arg.criteria,
                        }
                        for arg in a.args
                    ],
                }
                for a in level.required_actions
            ],
        }

    return {
        "id": _task_id_to_uuid7(task.task_id, version),
        "input": {
            "prompt": task.prompt,
            "task_id": task.task_id,
        },
        "expected_output": {
            "evaluation_criteria": {
                "minimal": task.evaluation_criteria.minimal,
                "optimal": task.evaluation_criteria.optimal,
            },
            "action_chain": {
                "minimal": action_chain_level_to_dict(task.action_chain.minimal),
                "optimal": action_chain_level_to_dict(task.action_chain.optimal),
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
