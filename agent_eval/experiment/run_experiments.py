"""
CLI entry point for running evaluations.

Usage:
    python -m agent_eval.experiment.run_experiments [OPTIONS]

Options:
    --evaluation-type TYPE   Run only this evaluation type (default: all)
    --dry-run                Build configs and validate tasks without running agents
    --no-validate            Skip resource pre-flight validation
    --nb-samples N           Limit number of tasks (for smoke tests)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import opik
from dotenv import load_dotenv
from opik.evaluation import evaluate

from agent_eval.benchmark.loader import build_run_configurations
from agent_eval.evaluators.opik.experiment_metrics import compute_experiment_metrics
from agent_eval.evaluators.opik.action_metric import ActionMetric
from agent_eval.evaluators.opik.result_accuracy_metric import ResultAccuracyMetric
from agent_eval.evaluators.opik.tool_call_stats_metric import ToolCallStatsMetric
from agent_eval.experiment.run_config import build_all_run_configs
from agent_eval.experiment.task import make_task
from agent_eval.tasks.loader import load_all_tasks, task_to_opik_item
from agent_eval.tasks.resource_validator import raise_on_errors, validate_all_tasks
from agent_eval.experiment.tracing import setup_tracing
from agent_eval.experiment.agent.code import ensure_docker_image

BENCHMARK_DIR = Path(__file__).parents[1] / "benchmark" / "config"
TASKS_DIR = Path(__file__).parents[1] / "tasks" / "config"
JUDGE_MODEL_PATH = (
    Path(__file__).parents[1] / "evaluators" / "core" / "config" / "judge_model.yml"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Metric builder ────────────────────────────────────────────────────────────


def get_scoring_metrics(metrics: list[str], judge_model_path: Path) -> list:
    selected = []
    selected.append(
        ResultAccuracyMetric(judge_model_path=judge_model_path)
    )  # includes efficiency + failure modes
    selected.append(ToolCallStatsMetric())  # literal tool-call efficiency tracking
    if "action_usage" in metrics:
        # ActionMetric runs the mapper once and emits usage + params + trajectory.
        selected.append(ActionMetric(judge_model_path=judge_model_path))
    return selected


# ── Opik dataset helper ───────────────────────────────────────────────────────


DATASET_NAME = "datagouv_tasks"


def get_or_create_dataset(client: opik.Opik, tasks) -> opik.Dataset:
    # Single shared dataset for the whole task set (item ids are derived from task_id,
    # see _task_id_to_uuid7). IMPORTANT: every dataset.insert() call writes a NEW
    # *version* of the WHOLE dataset (all current items get a version row), and Opik's
    # dataset-compare view fans out each result once per version ("Avg of N trials").
    # So we insert ONLY on first creation and skip entirely once the items exist —
    # any insert (even to add one task) re-versions every item and re-triggers the
    # fan-out. If the task set changes, regenerate the dataset cleanly (see --reset note).
    project_name = os.environ.get("OPIK_PROJECT_NAME")
    dataset = client.get_or_create_dataset(name=DATASET_NAME, project_name=project_name)
    items = [task_to_opik_item(task) for task in tasks]
    existing_ids = {item.get("id") for item in dataset.get_items()}
    missing = [item for item in items if item["id"] not in existing_ids]
    if not existing_ids:
        logger.info("Seeding %s with %d item(s)", DATASET_NAME, len(items))
        dataset.insert(items)
    elif missing:
        logger.warning(
            "%s already exists but %d task(s) are missing. NOT inserting (it would "
            "re-version every item and re-trigger the compare fan-out). Regenerate the "
            "dataset cleanly to pick up task changes.",
            DATASET_NAME, len(missing),
        )
    else:
        logger.info("Dataset %s already up to date (%d items)", DATASET_NAME, len(items))
    return dataset


def reset_dataset(client: opik.Opik, tasks) -> None:
    """Delete the dataset so the next seed starts at exactly one version per item.
    Use after adding/editing/removing tasks.

    The clean part is the API delete below. The ClickHouse cleanup it then calls is a
    local-Docker-specific workaround (see _clear_clickhouse_item_versions) — needed only
    because Opik's API delete does NOT cascade to `dataset_item_versions` and its compare
    view fans out across versions.
    """
    try:
        client.delete_dataset(name=DATASET_NAME)
        logger.info("Deleted dataset %s via API", DATASET_NAME)
    except Exception as exc:  # noqa: BLE001 - dataset may not exist yet
        logger.info("Dataset %s not deleted (%s) — continuing", DATASET_NAME, exc)

    item_ids = [task_to_opik_item(task)["id"] for task in tasks]
    _clear_clickhouse_item_versions(item_ids)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ DIRTY WORKAROUND — local self-hosted Opik (Docker) ONLY. SAFE TO DELETE.       ║
# ║                                                                                ║
# ║ Why it exists: Opik's dataset-compare view fans out each result once per       ║
# ║ dataset-item *version*, and deleting a dataset via the API does NOT remove its ║
# ║ `dataset_item_versions` rows from ClickHouse. Since our item ids are stable    ║
# ║ (derived from task_id), those orphaned versions keep re-triggering the fan-out ║
# ║ ("Avg of N trials") even after a fresh re-seed. ClickHouse isn't reachable     ║
# ║ from the host, so the only cleanup path here is `docker exec`.                 ║
# ║                                                                                ║
# ║ HOW TO REMOVE: delete this whole function and the `_clear_clickhouse_item_     ║
# ║ versions(item_ids)` call in reset_dataset above. Everything else stays clean.  ║
# ║ Remove it once Opik fixes the compare fan-out / cascades dataset deletes, or   ║
# ║ if you don't run Opik locally in Docker.                                       ║
# ║                                                                                ║
# ║ ENV VAR (optional): OPIK_CLICKHOUSE_CONTAINER — the ClickHouse container name. ║
# ║ Defaults to "opik-opik-clickhouse-1" (the standard local compose name), so you ║
# ║ normally do NOT need to set anything.                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _clear_clickhouse_item_versions(item_ids: list[str]) -> None:
    container = os.environ.get("OPIK_CLICKHOUSE_CONTAINER", "opik-opik-clickhouse-1")
    id_list = ",".join(f"'{item_id}'" for item_id in item_ids)
    query = (
        "ALTER TABLE opik.dataset_item_versions DELETE "
        f"WHERE dataset_item_id IN ({id_list}) SETTINGS mutations_sync=1"
    )
    try:
        subprocess.run(
            ["docker", "exec", container, "clickhouse-client", "--query", query],
            check=True, capture_output=True, text=True, timeout=120,
        )
        logger.info("Cleared ClickHouse item versions for %d task(s)", len(item_ids))
    except Exception as exc:  # noqa: BLE001 - docker/clickhouse may be unavailable
        logger.warning(
            "Could not clear ClickHouse dataset_item_versions automatically (%s). "
            "If you still see duplicated results, clear it manually: "
            'docker exec %s clickhouse-client --query '
            '"TRUNCATE TABLE opik.dataset_item_versions"',
            exc, container,
        )
# ── End dirty workaround ────────────────────────────────────────────────────────


# ── Experiment name ───────────────────────────────────────────────────────────


def experiment_name(run_config: dict) -> str:
    evaluation_type = run_config["evaluation_type"]
    model = run_config.get("model", {}).get("name")
    caps = "+".join(run_config.get("capabilities") or []) or "none"
    mcp_version = run_config.get("mcp_version")
    sp = run_config.get("system_prompt_name", "default")
    parts = ["datagouv", evaluation_type, model, caps]
    if mcp_version:
        parts.append(mcp_version)
    parts.append(sp)
    return "-".join(parts)


# ── Validation logging ────────────────────────────────────────────────────────


def _log_validation_summary(results: dict) -> None:
    for task_id, task_results in results.items():
        for r in task_results:
            level = logging.WARNING if r.severity == "error" else logging.INFO
            logger.log(
                level,
                "[Validation] %s | %s %s | %s: %s",
                task_id,
                r.resource_type,
                r.resource_id,
                r.check_fn,
                r.message,
            )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run datagouv-ai evaluations")
    parser.add_argument(
        "--evaluation-type",
        default=None,
        help="Run only this evaluation type. Omit to run all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build configs and validate tasks, but do not run agents.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip resource pre-flight validation.",
    )
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=None,
        help="Limit number of dataset items per experiment (smoke test).",
    )
    parser.add_argument(
        "--reset-dataset",
        action="store_true",
        help=(
            "Delete and re-seed the dataset (and clear its accumulated ClickHouse item "
            "versions) before running. Use after adding/editing/removing tasks."
        ),
    )
    args = parser.parse_args()

    load_dotenv(override=True)

    setup_tracing()

    # ── Load active tasks ─────────────────────────────────────────────────────
    tasks = load_all_tasks(TASKS_DIR)
    logger.info("Loaded %d active tasks", len(tasks))

    if not tasks:
        logger.error("No active tasks found in %s. Exiting.", TASKS_DIR)
        sys.exit(1)

    # ── Resource validation ───────────────────────────────────────────────────
    if not args.no_validate:
        tasks_to_validate = tasks[: args.nb_samples] if args.nb_samples else tasks
        logger.info(
            "Running resource validation on %d task(s)…", len(tasks_to_validate)
        )
        validation_results = asyncio.run(validate_all_tasks(tasks_to_validate))
        _log_validation_summary(validation_results)
        raise_on_errors(validation_results)  # sys.exit(1) on any mismatch
        logger.info("Resource validation passed.")

    # ── Build run configurations ──────────────────────────────────────────────
    run_configs_raw = build_run_configurations(BENCHMARK_DIR, args.evaluation_type)
    if not run_configs_raw:
        logger.error("No run configurations produced. Check benchmark YAMLs.")
        sys.exit(1)
    active_models = sorted({rc.model["name"] for rc in run_configs_raw})
    active_versions = sorted(
        {rc.mcp_version for rc in run_configs_raw if rc.mcp_version}
    )
    active_prompts = sorted({rc.system_prompt_name for rc in run_configs_raw})
    logger.info("Active models: %s", active_models)
    logger.info("Active MCP versions: %s", active_versions)
    logger.info("Active system prompts: %s", active_prompts)
    logger.info("Building %d run configurations…", len(run_configs_raw))
    run_configs = asyncio.run(build_all_run_configs(run_configs_raw))

    # ── Docker image pre-flight ───────────────────────────────────────────────
    if any("code" in (rc.get("capabilities") or []) for rc in run_configs):
        has_datagouv_cli = any(
            "datagouv-cli" in (rc.get("capabilities") or []) for rc in run_configs
        )
        ensure_docker_image(has_datagouv_cli=has_datagouv_cli)

    if args.dry_run:
        for rc in run_configs:
            logger.info("[DRY RUN] Would run: %s", experiment_name(rc))
        logger.info("[DRY RUN] Done — no agents were run.")
        return

    # ── Run evaluations ───────────────────────────────────────────────────────
    client = opik.Opik()
    if args.reset_dataset:
        reset_dataset(client, tasks)
    dataset = get_or_create_dataset(client, tasks)

    for run_config in run_configs:
        exp_name = experiment_name(run_config)
        logger.info("Starting experiment: %s", exp_name)

        scoring_metrics = get_scoring_metrics(run_config["metrics"], JUDGE_MODEL_PATH)

        evaluate(
            task_threads=1,
            dataset=dataset,
            task=make_task(run_config),
            scoring_metrics=scoring_metrics,
            experiment_scoring_functions=[compute_experiment_metrics],
            experiment_name=exp_name,
            experiment_config={
                "evaluation_type": run_config["evaluation_type"],
                "capabilities": run_config["capabilities"],
                "mcp_version": run_config.get("mcp_version"),
                "mcp_server_url": run_config.get("mcp_server_url"),
                "model": run_config["model"],
                "system_prompt_name": run_config["system_prompt_name"],
                "metrics": run_config["metrics"],
            },
            nb_samples=args.nb_samples,
            project_name=os.environ.get("OPIK_PROJECT_NAME"),
        )

        logger.info("Experiment complete: %s", exp_name)


if __name__ == "__main__":
    main()
