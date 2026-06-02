from __future__ import annotations

from pathlib import Path

import yaml

_SKILLS_DIR = Path(__file__).parents[2] / "benchmark" / "config" / "skills"
_VERSIONS_FILE = Path(__file__).parents[2] / "benchmark" / "config" / "skills_versions.yml"


def load_skills_prompt(version: str | None = None) -> str:
    """
    Load the skills context document for injection into the agent system prompt.
    If version is None, loads the latest active version.
    Returns empty string if no skills file is found.
    """
    with open(_VERSIONS_FILE, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    versions = [v for v in (raw.get("skills_versions") or []) if v.get("active")]
    if not versions:
        return ""

    if version:
        entry = next((v for v in versions if v.get("version") == version), None)
    else:
        entry = versions[-1]  # last active = latest

    if not entry:
        return ""

    skills_file = _SKILLS_DIR / entry["file_name"]
    if not skills_file.exists():
        return ""

    return skills_file.read_text(encoding="utf-8")
