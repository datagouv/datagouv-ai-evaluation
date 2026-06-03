"""
Semantic layer resolver: maps semantic action names + framework + version
to concrete tool names / endpoint patterns and framework-specific arg names.

Loaded from a directory containing actions.yml and action_args.yml (e.g. semantic_layer/config/).
Version matching uses PEP 440 specifiers via `packaging`
(e.g. "<=0.2.24", "==1", "" = any version).

Each tool/command/endpoint in actions.yml is an object {name, criteria?}.
criteria (optional): natural-language condition for LLM-based disambiguation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import SpecifierSet


def _version_matches(spec: str, version: str) -> bool:
    """Return True if *version* satisfies *spec*. Empty spec matches any version."""
    if not spec:
        return True
    try:
        return version in SpecifierSet(spec, prereleases=True)
    except Exception:
        return False


@dataclass
class ActionEntry:
    """A concrete tool/command/endpoint entry with optional disambiguation criteria."""

    name: str
    criteria: str | None = None


def _normalize_entry(item: Any) -> ActionEntry | None:
    """Normalize a tools/commands/endpoints list item to ActionEntry.
    Accepts strings ('tool_name') and dicts ({name: 'tool_name', criteria: '...'}).
    Returns None for empty strings or empty/invalid items.
    """
    if isinstance(item, str):
        return ActionEntry(name=item) if item else None
    if isinstance(item, dict):
        name = item.get("name", "")
        return ActionEntry(name=name, criteria=item.get("criteria")) if name else None
    return None


class SemanticLayerResolver:
    """
    Resolves semantic action names to concrete tool/command names and
    translates semantic arg names to framework-specific names.

    Usage:
        resolver = SemanticLayerResolver(Path("agent_eval/semantic_layer/config"))
        resolver.resolve_tool_entries("get.resource.profile", "mcp", "0.2.24")
        # → [ActionEntry(name="query_resource_data", criteria="page_size <= 5 ...")]
        resolver.resolve_tool_names("search.datasets", "mcp", "0.2.24")
        # → ["search_datasets"]
    """

    def __init__(self, semantic_layer_dir: Path) -> None:
        actions_path = semantic_layer_dir / "actions.yml"
        args_path = semantic_layer_dir / "action_args.yml"

        actions_raw: list[dict[str, Any]] = (
            yaml.safe_load(actions_path.read_text(encoding="utf-8")) or []
        )
        args_raw: dict[str, Any] = (
            yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        )

        # Build action index: {action_name: {framework: [{version, entries: [ActionEntry]}]}}
        self._actions: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for action_block in actions_raw:
            if not isinstance(action_block, dict):
                continue
            for action_name, action_def in action_block.items():
                if not isinstance(action_def, dict):
                    continue
                entry: dict[str, list[dict[str, Any]]] = {}
                for framework in ("mcp", "api", "cli"):
                    fw_entries = action_def.get(framework) or []
                    if fw_entries:
                        entry[framework] = fw_entries
                self._actions[action_name] = entry

        # Build arg index: {semantic_arg: {framework: [{version, name}]}}
        self._args: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for arg_name, fw_list in args_raw.items():
            if not isinstance(fw_list, list):
                continue
            arg_entry: dict[str, list[dict[str, Any]]] = {}
            for fw_item in fw_list:
                if not isinstance(fw_item, dict):
                    continue
                for fw, fw_def in fw_item.items():
                    if not isinstance(fw_def, dict):
                        continue
                    arg_entry.setdefault(fw, []).append(fw_def)
            self._args[arg_name] = arg_entry

    def resolve_tool_entries(
        self, action: str, framework: str, version: str
    ) -> list[ActionEntry]:
        """
        Return all ActionEntry objects (name + optional criteria) for
        (action, framework, version). Returns [] if no match.
        """
        fw_entries = self._actions.get(action, {}).get(framework)
        if not fw_entries:
            return []
        key = (
            "tools"
            if framework == "mcp"
            else "commands"
            if framework == "cli"
            else "endpoints"
        )
        for entry in fw_entries:
            if _version_matches(str(entry.get("version", "")), version):
                raw_items = entry.get(key) or []
                return [
                    e for item in raw_items if (e := _normalize_entry(item)) is not None
                ]
        return []

    def resolve_tool_names(
        self, action: str, framework: str, version: str
    ) -> list[str]:
        """
        Return concrete tool/command names for (action, framework, version).
        Returns [] if the action is unknown or has no entry for this framework/version.
        Backward-compatible wrapper over resolve_tool_entries().
        """
        return [e.name for e in self.resolve_tool_entries(action, framework, version)]

    def action_names(self) -> list[str]:
        """Return all known semantic action names."""
        return list(self._actions.keys())

    def resolve_arg_name(self, semantic_arg: str, framework: str, version: str) -> str:
        """
        Map a semantic arg name to the framework-specific name for the given version.
        Returns semantic_arg unchanged if no mapping is found.
        """
        fw_entries = self._args.get(semantic_arg, {}).get(framework)
        if not fw_entries:
            return semantic_arg
        for entry in fw_entries:
            if _version_matches(str(entry.get("version", "")), version):
                name = entry.get("name")
                if name:
                    return str(name)
        return semantic_arg
