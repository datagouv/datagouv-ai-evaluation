import json
from pathlib import Path

import pytest
import logging

from mcp_eval.agent.agent import run_agent
from mcp_eval.metrics.deterministic import score_deterministic
from mcp_eval.metrics.judge import score_with_llm_judge

DATASET_PATH = Path(__file__).parents[1] / "datasets" / "benchmark.jsonl"

logger = logging.getLogger(__name__)


def load_dataset():
    tasks = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


@pytest.mark.anyio
async def test_benchmark_suite():

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    tasks = load_dataset()

    failures = []
    for task in tasks:
        # helpful debug
        logger.info(f"RUN {task['task_id']} model={task['model']}")

        result = await run_agent(task)
        det = score_deterministic(result)
        judge = await score_with_llm_judge(task["prompt"], result)

        if not det.answer_present:
            failures.append(
                (
                    task["task_id"],
                    "no_answer",
                    result.answer[:200] if result.answer else "",
                )
            )

        if judge and judge.score < 0.3:
            failures.append((task["task_id"], f"judge<{0.3}", judge.rationale[:200]))

    assert not failures, f"Benchmark failures: {failures}"
