import json
import pandas as pd
from pathlib import Path

import opik
from dotenv import load_dotenv

load_dotenv(override=True)

EXAMPLE_PATH = Path(__file__).parents[1] / "datasets" / "dataset.jsonl"


def read_jsonl(path: Path) -> pd.DataFrame:
    with open(path) as f:
        lines = f.read().splitlines()
    return pd.DataFrame([json.loads(line) for line in lines])


def build_opik_items(df: pd.DataFrame) -> list[dict]:
    items = []
    for _, row in df.iterrows():
        items.append({
            "input": {
                "prompt": row["prompt"],
            },
            "expected_output": {
                "expected_tool_calls": row["expected_tool_calls"],
            },
            "metadata": {
                "tested_tools": row["tested_tools"],
                "difficulty": row["difficulty"],
                "turn": row["turn"],
            },
        })
    return items


df_examples = read_jsonl(EXAMPLE_PATH)

client = opik.Opik()

dataset = client.get_or_create_dataset(
    name="eval_datagouv_mcp",
    description="MCP benchmark dataset for datagouv evaluation",
)
dataset.insert(build_opik_items(df_examples))

print(f"Dataset ready: {dataset.name} ({len(df_examples)} items)")
