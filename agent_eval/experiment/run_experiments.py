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

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from opik.evaluation import evaluate

import opik
from agent_eval.benchmark.loader import build_run_configurations
from agent_eval.evaluators.opik.action_metric import ActionMetric
from agent_eval.evaluators.opik.experiment_metrics import compute_experiment_metrics
from agent_eval.evaluators.opik.result_accuracy_metric import ResultAccuracyMetric
from agent_eval.evaluators.opik.tool_call_stats_metric import ToolCallStatsMetric
from agent_eval.experiment.agent.code import ensure_docker_image
from agent_eval.experiment.run_config import build_all_run_configs
from agent_eval.experiment.task import make_task
from agent_eval.experiment.tracing import setup_tracing
from agent_eval.tasks.loader import load_all_tasks, task_to_opik_item
from agent_eval.tasks.resource_validator import raise_on_errors, validate_all_tasks

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


# ── Dataset version ───────────────────────────────────────────────────────────
# BUMP THIS WHENEVER YOU ADD/EDIT/REMOVE A TASK to get a clean fresh dataset.
# The version is folded into the dataset name AND each item id, so a new version
# produces entirely fresh ids that don't collide with orphaned `dataset_item_versions`
# rows from prior submissions. Within one version, re-runs reuse the same dataset and
# skip insert (no version churn). If you forget to bump after editing a task, the
# dataset stays on the old content — see the warning logged in get_or_create_dataset.
DATASET_VERSION = "v3"

DATASET_NAME_BASE = "datagouv_tasks"


def _dataset_name() -> str:
    return f"{DATASET_NAME_BASE}_{DATASET_VERSION}"


def get_or_create_dataset(client: opik.Opik, tasks, project_name: str) -> opik.Dataset:
    """Get or seed the versioned dataset.

    The single seed pattern (insert ONLY when empty) keeps every item at exactly one
    `dataset_item_versions` row, which is what stops Opik's dataset-compare view from
    fanning out results ("Avg of N trials"). To apply task changes, bump DATASET_VERSION
    — that yields a fresh dataset name + fresh item ids, no collision with orphans.
    """
    name = _dataset_name()
    dataset = client.get_or_create_dataset(name=name, project_name=project_name)
    items = [task_to_opik_item(task, version=DATASET_VERSION) for task in tasks]
    existing_ids = {item.get("id") for item in dataset.get_items()}
    missing = [item for item in items if item["id"] not in existing_ids]
    if not existing_ids:
        logger.info("Seeding %s with %d item(s)", name, len(items))
        dataset.insert(items)
    elif missing:
        logger.warning(
            "%s already exists but %d task(s) are missing. NOT inserting (it would "
            "re-version every item and re-trigger the compare fan-out). Bump "
            "DATASET_VERSION in run_experiments.py to pick up task changes cleanly.",
            name,
            len(missing),
        )
    else:
        logger.info("Dataset %s already up to date (%d items)", name, len(items))
    return dataset


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
    args = parser.parse_args()

    load_dotenv(override=True)

    project_name = os.environ.get("OPIK_PROJECT_NAME")
    if not project_name:
        logger.error(
            "OPIK_PROJECT_NAME is not set. Add it to your .env file "
            "(e.g. OPIK_PROJECT_NAME=datagouv-ai-evaluation)."
        )
        sys.exit(1)
    logger.info("Opik project: %s", project_name)

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
    dataset = get_or_create_dataset(client, tasks, project_name)

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
        )

        logger.info("Experiment complete: %s", exp_name)


if __name__ == "__main__":
    main()
