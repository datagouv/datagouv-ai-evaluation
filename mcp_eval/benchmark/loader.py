from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class MCPVersionConfig:
    version: str
    active: bool
    server_url: str


@dataclass
class EvaluationTypeConfig:
    name: str
    description: str
    capabilities: list[list[str]]  # e.g. [[], ["mcp"], ["mcp", "web"]]
    metrics: list[str]             # e.g. ["result_accuracy", "efficiency", "tool_usage"]
    system_prompts_filter: list[str] | str = "active"  # list of names | "active" | "all"


@dataclass
class SystemPromptConfig:
    name: str
    active: bool
    prompt: str


@dataclass
class RunConfiguration:
    """A single row of the cross-product: one concrete experiment to run."""
    evaluation_type: str
    capabilities: list[str]         # e.g. ["mcp", "web"]
    mcp_version: str | None         # None for data_contamination (no MCP server)
    mcp_server_url: str | None
    model: str
    system_prompt_name: str
    system_prompt: str
    metrics: list[str] = field(default_factory=list)


# ── YAML loaders ──────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_evaluation_types(path: Path) -> list[EvaluationTypeConfig]:
    raw = _load_yaml(path)
    configs = []
    for entry in raw.get("evaluation_types") or []:
        # The YAML format uses the name as a sibling key with None value:
        #   - my_type:         ← name key, value is None
        #     description: …
        #     capabilities: …
        # We identify the name as the key whose value is None (or not a known field).
        if not isinstance(entry, dict):
            continue
        known_fields = {"description", "capabilities", "metrics", "system_prompts"}
        name = next(
            (k for k, v in entry.items() if k not in known_fields),
            None,
        )
        if name is None:
            continue
        sp_filter = entry.get("system_prompts", "active")
        configs.append(EvaluationTypeConfig(
            name=name,
            description=(entry.get("description") or "").strip(),
            capabilities=entry.get("capabilities") or [[]],
            metrics=entry.get("metrics") or [],
            system_prompts_filter=sp_filter,
        ))
    return configs


def load_mcp_versions(path: Path) -> list[MCPVersionConfig]:
    raw = _load_yaml(path)
    return [
        MCPVersionConfig(
            version=str(e["version"]),
            active=bool(e.get("active", True)),
            server_url=str(e["server_url"]),
        )
        for e in (raw.get("mcp_versions") or [])
        if e.get("active", True)
    ]


def load_models(path: Path) -> list[str]:
    """Return a flat list of model name strings (provider:model format)."""
    raw = _load_yaml(path)
    names = []
    for provider_block in raw.get("models") or []:
        for entry in provider_block.get("entries") or []:
            if entry.get("name") and entry.get("active", True):
                names.append(entry["name"])
    return names


def load_system_prompts(path: Path) -> list[SystemPromptConfig]:
    raw = _load_yaml(path)
    configs = []
    for entry in raw.get("system_prompts") or []:
        prompt_text = (entry.get("prompt") or "").strip()
        configs.append(SystemPromptConfig(
            name=entry["name"],
            active=bool(entry.get("active", True)),
            prompt=prompt_text,
        ))
    return configs


# ── Cross-product builder ─────────────────────────────────────────────────────

def _resolve_system_prompts(
    all_sps: list[SystemPromptConfig],
    filt: list[str] | str,
) -> list[SystemPromptConfig]:
    """Return the subset of system prompts selected by *filt*."""
    if isinstance(filt, list):
        by_name = {sp.name: sp for sp in all_sps}
        return [by_name[n] for n in filt if n in by_name]
    if filt == "all":
        return list(all_sps)
    # default: "active"
    return [sp for sp in all_sps if sp.active]


def build_run_configurations(
    benchmark_dir: Path,
    evaluation_type_filter: str | None = None,
) -> list[RunConfiguration]:
    """
    Load all benchmark YAMLs and produce the cross-product of run configurations.

    Cross-product logic:
    - data_contamination: capabilities=[[]] → one entry per (model, system_prompt), no MCP server
    - mcp_versions: capabilities=[["mcp"]] → iterate mcp_versions × models × system_prompts
    - capabilities_benchmark: all capabilities combos × mcp_versions × models × system_prompts
    """
    eval_types = load_evaluation_types(benchmark_dir / "evaluation_types.yml")
    mcp_versions = load_mcp_versions(benchmark_dir / "mcp_versions.yml")
    models = load_models(benchmark_dir / "models.yml")
    system_prompts = load_system_prompts(benchmark_dir / "system_prompts.yml")

    if not models:
        raise ValueError(
            f"No models found in {benchmark_dir / 'models.yml'}. "
            "Please populate models.yml before running evaluation."
        )

    runs: list[RunConfiguration] = []

    for et in eval_types:
        if evaluation_type_filter and et.name != evaluation_type_filter:
            continue

        et_system_prompts = _resolve_system_prompts(system_prompts, et.system_prompts_filter)

        for capabilities in et.capabilities:
            uses_mcp = "mcp" in capabilities

            if uses_mcp:
                mcp_iter = mcp_versions
            else:
                # data_contamination and similar: one synthetic "no server" entry
                mcp_iter = [None]

            for mcp_cfg in mcp_iter:
                for model in models:
                    for sp in et_system_prompts:
                        runs.append(RunConfiguration(
                            evaluation_type=et.name,
                            capabilities=list(capabilities),
                            mcp_version=mcp_cfg.version if mcp_cfg else None,
                            mcp_server_url=mcp_cfg.server_url if mcp_cfg else None,
                            model=model,
                            system_prompt_name=sp.name,
                            system_prompt=sp.prompt,
                            metrics=list(et.metrics),
                        ))

    return runs
