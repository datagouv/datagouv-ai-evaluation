import json
import pandas as pd
from pathlib import Path
from phoenix.client import Client
from typing import Any
from mcp_eval.experiment.run_config import RunConfig

EXAMPLE_PATH = Path(__file__).parents[1] / "datasets" / "dataset.jsonl"
CONFIG_PATH = Path(__file__).parents[1] / "datasets" / "config.json"


def read_jsonl(path: Path) -> pd.DataFrame:
    lines = []
    with open(path) as f:
        lines = f.read().splitlines()

    line_dicts = [json.loads(line) for line in lines]
    return pd.DataFrame(line_dicts)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


df_examples = read_jsonl(EXAMPLE_PATH)
config = read_json(CONFIG_PATH)

df_run_config = RunConfig(
    mcp_versions=config["mcp_versions"],
    models=config["models"],
    system_prompts=config["system_prompts"],
).config

client = Client(base_url="http://localhost:6006")

dataset = client.datasets.create_dataset(
    name="eval_datagouv_mcp",
    dataframe=df_examples,
    input_keys=["prompt"],
    output_keys=["expected_tool_calls"],
)
