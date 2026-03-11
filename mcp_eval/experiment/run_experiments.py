import json
from pathlib import Path
from typing import Any
import asyncio

from phoenix.client import Client, AsyncClient

from mcp_eval.experiment.run_config import RunConfig, build_run_config_df
from mcp_eval.experiment.task import make_task
from mcp_eval.evaluators.tool_selection_match import tool_selection_match
from mcp_eval.evaluators.tool_invocation_judge import tool_invocation_correctness_judge

CONFIG_PATH = Path(__file__).parents[1] / "datasets" / "config.json"


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def main():
    config = read_json(CONFIG_PATH)

    run_config_obj = RunConfig(
        mcp_versions=config["mcp_versions"],
        models=config["models"],
        system_prompts=config["system_prompts"],
    )
    df_run_config = await build_run_config_df(run_config_obj)

    client = Client(base_url="http://localhost:6006")
    dataset = client.datasets.get_dataset(dataset="eval_datagouv_mcp")

    async_client = AsyncClient(base_url="http://localhost:6006")

    for _, row in df_run_config.iterrows():
        run_config = row.to_dict()
        task = make_task(run_config)

        experiment = await async_client.experiments.run_experiment(
            # dry_run=1,
            dataset=dataset,
            task=task,
            evaluators=[tool_selection_match, tool_invocation_correctness_judge],
            experiment_name=f"datagouv-mcp-{run_config['mcp_version']}-{run_config['model']}",
            experiment_description="MCP benchmark over dataset.jsonl",
            experiment_metadata={
                "mcp_version": run_config["mcp_version"],
                "mcp_server_url": run_config["mcp_server_url"],
                "mcp_tools": run_config["mcp_tools_description"],
                "model": run_config["model"],
                "system_prompt": run_config["system_prompt"],
            },
        )

        print(experiment)


if __name__ == "__main__":
    asyncio.run(main())
