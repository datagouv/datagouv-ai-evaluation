import json
import pandas as pd
from pathlib import Path
from phoenix.client import Client

EXAMPLE_PATH = Path(__file__).parents[1] / "datasets" / "dataset.jsonl"


def read_jsonl(path: Path) -> pd.DataFrame:
    lines = []
    with open(path) as f:
        lines = f.read().splitlines()

    line_dicts = [json.loads(line) for line in lines]
    return pd.DataFrame(line_dicts)


df_examples = read_jsonl(EXAMPLE_PATH)

client = Client(base_url="http://localhost:6006")

dataset = client.datasets.create_dataset(
    name="eval_datagouv_mcp",
    dataframe=df_examples,
    split_keys=["tested_tools", "difficulty", "turn"],
    input_keys=["prompt"],
    output_keys=["expected_tool_calls"],
)
