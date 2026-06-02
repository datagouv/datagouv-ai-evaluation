"""Project-wide .env loader.

Resolves the .env path relative to the project root (two levels above this file)
so imports work regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

_PROJECT_ROOT = Path(__file__).parent.parent
ENV_VALUES: dict[str, str | None] = dotenv_values(_PROJECT_ROOT / ".env")
