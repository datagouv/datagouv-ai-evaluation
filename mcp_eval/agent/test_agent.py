import json
from pathlib import Path
import asyncio
from mcp_eval.agent.agent import run_agent

DATASET_PATH = Path(__file__).parents[1] / "datasets" / "benchmark.jsonl"


def load_dataset():
    tasks = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


tasks = load_dataset()

for task in tasks[0:1]:
    result = asyncio.run(run_agent(task))
