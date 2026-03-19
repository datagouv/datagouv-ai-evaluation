from dataclasses import dataclass
import pandas as pd
from typing import Literal
from mcp_eval.experiment.mcp_tools_getter import get_mcp_tools


@dataclass
class MCPVersion:
    mcp_version: str
    mcp_server_url: str


@dataclass
class RunConfig:
    mcp_versions: list[MCPVersion]
    models: list[Literal["mistral:mistral-medium-latest"]]
    system_prompts: list[str]


async def build_run_config_df(run_config: RunConfig) -> pd.DataFrame:
    df_servers = pd.DataFrame(run_config.mcp_versions)

    tool_descriptions, tool_names = [], []
    for url in df_servers["mcp_server_url"]:
        description, names = await get_mcp_tools(url)
        tool_descriptions.append(description)
        tool_names.append(names)

    df_servers["mcp_tools_description"] = tool_descriptions
    df_servers["mcp_tool_names"] = tool_names

    df_models = pd.DataFrame({"model": run_config.models})
    df_prompts = pd.DataFrame({"system_prompt": run_config.system_prompts})

    return df_servers.merge(df_models, how="cross").merge(df_prompts, how="cross")


mcp_versions = [
    MCPVersion(**{"mcp_version": "1", "mcp_server_url": "https://mcp.data.gouv.fr/mcp"})
]
models = ["mistral:mistral-medium-latest"]
system_prompts = [
    "You are a data assistant using an MCP server. Use MCP tools when relevant to retrieve factual information. Provide a final answer to the user."
]
