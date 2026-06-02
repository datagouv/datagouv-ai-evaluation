from __future__ import annotations

import json
import logging
import string
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

import httpx

from agent_eval.tasks.loader import Task

logger = logging.getLogger(__name__)

# Base URL for the data.gouv.fr REST API (never use MCP tools here — they may change)
_API_BASE = "https://www.data.gouv.fr/api/1"

_API_TABULAR = "https://tabular-api.data.gouv.fr"

# Templates: paths relative to _API_BASE, or full URLs for other bases.
# Placeholder names drive validation: {id} = resource_id, {rid} = resource_id,
# {dataset_id} = dataset_id (required on the Resource YAML entry).
_TYPE_ENDPOINTS: dict[str, str] = {
    "dataset": "/datasets/{id}",
    "resource": "/datasets/{dataset_id}/resources/{rid}",
    "load_resource": "/datasets/r/{rid}",
    "dataservice": "/dataservices/{id}",
    "organization": "/organizations/{id}",
    "tabular_api": _API_TABULAR + "/api/resources/{id}/",
}


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    resource_id: str
    resource_type: str
    check_fn: str
    passed: bool
    message: str
    severity: Literal["info", "error"]


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _construct_url(
    resource_type: str,
    resource_id: str,
    dataset_id: str | None = None,
) -> str:
    """
    Build and validate the full URL for a given resource type and id.
    Parses the template placeholders and raises ValueError for missing required params.
    Templates that start with 'http' are used as-is (absolute URL from another base).
    """
    template = _TYPE_ENDPOINTS.get(resource_type)
    if template is None:
        raise ValueError(f"Unknown resource type: {resource_type!r}")

    placeholders = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    }

    kwargs: dict[str, str] = {}
    if "dataset_id" in placeholders:
        if not dataset_id:
            raise ValueError(
                f"Resource type {resource_type!r} requires a dataset_id "
                f"(resource {resource_id} has none set)"
            )
        kwargs["dataset_id"] = dataset_id
    if "rid" in placeholders:
        kwargs["rid"] = resource_id
    if "id" in placeholders:
        kwargs["id"] = resource_id

    path = template.format(**kwargs)
    return path if path.startswith("http") else f"{_API_BASE}{path}"


async def _get_json(client: httpx.AsyncClient, url: str) -> dict:
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response.json()


async def _fetch_metadata(
    client: httpx.AsyncClient,
    resource_type: str,
    resource_id: str,
    dataset_id: str | None = None,
) -> dict:
    """Fetch JSON metadata from the data.gouv.fr REST API."""
    url = _construct_url(resource_type, resource_id, dataset_id)
    return await _get_json(client, url)


# ── Snapshot helpers ──────────────────────────────────────────────────────────


def _snapshot_path(snapshot_dir: Path, task_id: str, resource_id: str) -> Path:
    safe_rid = resource_id.replace("-", "_")
    return snapshot_dir / f"{task_id}_{safe_rid}.json"


def _read_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_snapshot(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Check functions ───────────────────────────────────────────────────────────


async def same_title_description(
    resource_type: str,
    resource_id: str,
    snapshot_dir: Path,
    task_id: str,
    dataset_id: str | None = None,
) -> ValidationResult:
    """
    Fetch current title+description from data.gouv.fr REST API and compare with snapshot.
    - No snapshot → write it, return info (passed=True)
    - Match → passed=True
    - Mismatch → passed=False, severity=error (caller must block evaluation)
    """
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            meta = await _fetch_metadata(client, resource_type, resource_id, dataset_id)
        except Exception as exc:
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="same_title_description",
                passed=False,
                message=f"API fetch failed: {exc}",
                severity="error",
            )

    current = {
        "title": meta.get("title") or meta.get("name") or "",
        "description": meta.get("description") or "",
    }

    snap_path = _snapshot_path(snapshot_dir, task_id, resource_id)
    stored = _read_snapshot(snap_path)

    if stored is None:
        _write_snapshot(snap_path, current)
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="same_title_description",
            passed=True,
            message=f"Snapshot created at {snap_path}",
            severity="info",
        )

    if stored == current:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="same_title_description",
            passed=True,
            message="Title and description match snapshot",
            severity="info",
        )

    return ValidationResult(
        resource_id=resource_id,
        resource_type=resource_type,
        check_fn="same_title_description",
        passed=False,
        message=(
            f"Title/description changed for {resource_type} {resource_id}.\n"
            f"  Stored : title={stored['title']!r}, description={stored['description'][:80]!r}\n"
            f"  Current: title={current['title']!r}, description={current['description'][:80]!r}\n"
            f"  If intentional, delete {snap_path} and re-run to accept the new values."
        ),
        severity="error",
    )


async def download_url_reachable(
    resource_type: str, resource_id: str, dataset_id: str | None = None
) -> ValidationResult:
    """Check that the resource's download URL is reachable via the data.gouv.fr redirect."""
    try:
        url = _construct_url("load_resource", resource_id, dataset_id)
    except ValueError as exc:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="download_url_reachable",
            passed=False,
            message=str(exc),
            severity="error",
        )

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.head(url, follow_redirects=True)
            if resp.is_success or resp.status_code == 405:
                return ValidationResult(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    check_fn="download_url_reachable",
                    passed=True,
                    message=f"URL reachable: {url} (HTTP {resp.status_code})",
                    severity="info",
                )
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="download_url_reachable",
                passed=False,
                message=f"URL returned HTTP {resp.status_code}: {url}",
                severity="error",
            )
        except Exception as exc:
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="download_url_reachable",
                passed=False,
                message=f"URL unreachable: {url} — {exc}",
                severity="error",
            )


async def tabular_api_available(
    resource_type: str, resource_id: str, **_
) -> ValidationResult:
    """Check that the tabular API endpoint for this resource responds."""
    url = _construct_url("tabular_api", resource_id)
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url, follow_redirects=True)
        except Exception as exc:
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="tabular_api_available",
                passed=False,
                message=f"Tabular API unreachable: {url} — {exc}",
                severity="error",
            )

    if resp.is_success:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="tabular_api_available",
            passed=True,
            message=f"Tabular API available (HTTP {resp.status_code}): {url}",
            severity="info",
        )
    return ValidationResult(
        resource_id=resource_id,
        resource_type=resource_type,
        check_fn="tabular_api_available",
        passed=False,
        message=f"Tabular API returned HTTP {resp.status_code}: {url}",
        severity="error",
    )


_RELATIVE_DELTAS: dict[str, timedelta] = {
    "last_7_days": timedelta(days=7),
    "last_30_days": timedelta(days=30),
    "last_90_days": timedelta(days=90),
    "last_365_days": timedelta(days=365),
}


async def update_date_more_recent_than(
    resource_type: str,
    resource_id: str,
    type: str = "relative",
    value: str = "last_30_days",
    dataset_id: str | None = None,
) -> ValidationResult:
    """
    Check that the resource was last modified more recently than the given threshold.
    type='relative', value='last_7_days'  → must have been updated within the past 7 days
    type='absolute', value='2024-01-01'   → must have been updated after that ISO date
    """
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            meta = await _fetch_metadata(client, resource_type, resource_id, dataset_id)
        except Exception as exc:
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="update_date_more_recent_than",
                passed=False,
                message=f"API fetch failed: {exc}",
                severity="error",
            )

    last_modified_raw = meta.get("last_modified") or meta.get("internal", {}).get(
        "last_modified_internal"
    )
    if not last_modified_raw:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="update_date_more_recent_than",
            passed=False,
            message="No last_modified field found in resource metadata",
            severity="error",
        )

    try:
        last_modified = datetime.fromisoformat(
            str(last_modified_raw).rstrip("Z")
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="update_date_more_recent_than",
            passed=False,
            message=f"Cannot parse last_modified date {last_modified_raw!r}: {exc}",
            severity="error",
        )

    now = datetime.now(tz=timezone.utc)
    if type == "relative":
        delta = _RELATIVE_DELTAS.get(value)
        if delta is None:
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="update_date_more_recent_than",
                passed=False,
                message=f"Unknown relative value {value!r}. Use one of: {list(_RELATIVE_DELTAS)}",
                severity="error",
            )
        threshold = now - delta
    elif type == "absolute":
        try:
            threshold = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            return ValidationResult(
                resource_id=resource_id,
                resource_type=resource_type,
                check_fn="update_date_more_recent_than",
                passed=False,
                message=f"Cannot parse absolute threshold date {value!r}: {exc}",
                severity="error",
            )
    else:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="update_date_more_recent_than",
            passed=False,
            message=f"Unknown type {type!r}. Use 'relative' or 'absolute'",
            severity="error",
        )

    if last_modified >= threshold:
        return ValidationResult(
            resource_id=resource_id,
            resource_type=resource_type,
            check_fn="update_date_more_recent_than",
            passed=True,
            message=f"Last modified {last_modified.date()} is more recent than threshold {threshold.date()}",
            severity="info",
        )
    return ValidationResult(
        resource_id=resource_id,
        resource_type=resource_type,
        check_fn="update_date_more_recent_than",
        passed=False,
        message=(
            f"Resource not updated recently enough: last modified {last_modified.date()}, "
            f"threshold {threshold.date()} ({type}={value})"
        ),
        severity="error",
    )


# ── Registry & orchestration ──────────────────────────────────────────────────

CHECK_REGISTRY: dict[str, Callable] = {
    "same_title_description": same_title_description,
    "download_url_reachable": download_url_reachable,
    "tabular_api_available": tabular_api_available,
    "update_date_more_recent_than": update_date_more_recent_than,
}


async def validate_task(task: Task, snapshot_dir: Path) -> list[ValidationResult]:
    """Run all resource checks for a single task."""
    import asyncio

    coros = []
    for resource in task.resources:
        for check in resource.checks:
            fn = CHECK_REGISTRY.get(check.fn)
            if fn is None:
                logger.warning(
                    "Unknown check function %r for task %s", check.fn, task.task_id
                )
                continue

            if check.fn == "same_title_description":
                coros.append(
                    fn(
                        resource.type,
                        resource.id,
                        snapshot_dir,
                        task.task_id,
                        dataset_id=resource.dataset_id,
                    )
                )
            else:
                coros.append(
                    fn(
                        resource.type,
                        resource.id,
                        dataset_id=resource.dataset_id,
                        **check.params,
                    )
                )

    if not coros:
        return []
    return list(await asyncio.gather(*coros))


async def validate_all_tasks(
    tasks: list[Task],
    snapshot_dir: Path | None = None,
) -> dict[str, list[ValidationResult]]:
    """
    Run resource validation for all tasks.
    snapshot_dir defaults to agent_eval/tasks/data/.
    """
    import asyncio

    if snapshot_dir is None:
        snapshot_dir = Path(__file__).parent / "data"

    coros = [validate_task(task, snapshot_dir) for task in tasks]
    task_results = await asyncio.gather(*coros)
    return {task.task_id: list(results) for task, results in zip(tasks, task_results)}


def raise_on_errors(results: dict[str, list[ValidationResult]]) -> None:
    """
    If any error-severity validation failures exist, print a report and exit with code 1.
    Humans must resolve each error (e.g. delete the stale snapshot file) before re-running.
    """
    errors: list[tuple[str, ValidationResult]] = [
        (task_id, r)
        for task_id, task_results in results.items()
        for r in task_results
        if r.severity == "error" and not r.passed
    ]

    if not errors:
        return

    lines = ["\n[Resource Validation] Evaluation blocked — manual review required:\n"]
    for task_id, r in errors:
        lines.append(
            f"  Task {task_id} | {r.resource_type} {r.resource_id} | {r.check_fn}"
        )
        lines.append(f"    {r.message}\n")

    print("\n".join(lines), file=sys.stderr)
    sys.exit(1)
