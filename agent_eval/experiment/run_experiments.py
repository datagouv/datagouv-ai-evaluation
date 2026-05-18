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
import sys
from pathlib import Path

import opik
from dotenv import load_dotenv
from opik.evaluation import evaluate

from agent_eval.benchmark.loader import build_run_configurations
from agent_eval.evaluators.opik.experiment_metrics import compute_experiment_metrics
from agent_eval.evaluators.opik.result_accuracy_metric import ResultAccuracyMetric
from agent_eval.evaluators.opik.tool_usage_metric import ToolUsageMetric
from agent_eval.evaluators.opik.trajectory_metric import TrajectoryAdherenceMetric
from agent_eval.experiment.run_config import build_all_run_configs
from agent_eval.experiment.task import make_task
from agent_eval.tasks.loader import load_all_tasks, task_to_opik_item
from agent_eval.tasks.resource_validator import raise_on_errors, validate_all_tasks
from agent_eval.experiment.tracing import setup_tracing

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
    if "tool_usage" in metrics:
        selected.append(ToolUsageMetric(judge_model_path=judge_model_path))
        selected.append(TrajectoryAdherenceMetric(judge_model_path=judge_model_path))
    return selected


# ── Opik dataset helper ───────────────────────────────────────────────────────


def get_or_create_dataset(
    client: opik.Opik, tasks, evaluation_type: str
) -> opik.Dataset:
    name = f"datagouv_mcp_{evaluation_type}"
    dataset = client.get_or_create_dataset(name=name)
    items = [task_to_opik_item(task) for task in tasks]
    dataset.insert(items)
    return dataset


# ── Experiment name ───────────────────────────────────────────────────────────


def experiment_name(run_config: dict) -> str:
    caps = "+".join(run_config.get("capabilities") or []) or "none"
    version = run_config.get("mcp_version") or "no-tools"
    model = run_config.get("model", {}).get("name")
    sp = run_config.get("system_prompt_name", "default")
    return f"datagouv-{run_config['evaluation_type']}-{version}-{model}-{caps}-{sp}"


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

    if args.dry_run:
        for rc in run_configs:
            logger.info("[DRY RUN] Would run: %s", experiment_name(rc))
        logger.info("[DRY RUN] Done — no agents were run.")
        return

    # ── Run evaluations ───────────────────────────────────────────────────────
    client = opik.Opik()

    for run_config in run_configs:
        exp_name = experiment_name(run_config)
        logger.info("Starting experiment: %s", exp_name)

        dataset = get_or_create_dataset(client, tasks, run_config["evaluation_type"])
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
