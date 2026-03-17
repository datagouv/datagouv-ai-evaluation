import json
from pathlib import Path
from typing import Any
import asyncio

import opik
from opik.evaluation import evaluate
from dotenv import load_dotenv

from mcp_eval.experiment.run_config import RunConfig, build_run_config_df
from mcp_eval.experiment.task import make_task
from mcp_eval.evaluators.tool_selection_match import ToolSelectionMatch
from mcp_eval.evaluators.tool_invocation_judge import ToolInvocationCorrectnessJudge

load_dotenv(override=True)

CONFIG_PATH = Path(__file__).parents[1] / "datasets" / "config.json"


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    config = read_json(CONFIG_PATH)

    run_config_obj = RunConfig(
        mcp_versions=config["mcp_versions"],
        models=config["models"],
        system_prompts=config["system_prompts"],
    )
    df_run_config = asyncio.run(build_run_config_df(run_config_obj))

    client = opik.Opik()
    dataset = client.get_dataset(name="eval_datagouv_mcp")

    for _, row in df_run_config.iterrows():
        run_config = row.to_dict()
        task = make_task(run_config)

        evaluate(
            # nb_samples=1,  # décommenter pour un smoke test rapide
            task_threads=1,
            dataset=dataset,
            task=task,
            scoring_metrics=[ToolSelectionMatch(), ToolInvocationCorrectnessJudge()],
            experiment_name=f"datagouv-mcp-{run_config['mcp_version']}-{run_config['model']}",
            experiment_config={
                "mcp_version": run_config["mcp_version"],
                "mcp_server_url": run_config["mcp_server_url"],
                "mcp_tools": run_config["mcp_tools_description"],
                "model": run_config["model"],
                "system_prompt": run_config["system_prompt"],
            },
        )


if __name__ == "__main__":
    main()
