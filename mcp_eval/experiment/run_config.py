from dataclasses import dataclass
import pandas as pd
from typing import Literal


@dataclass
class MCPVersion:
    mcp_version: str
    mcp_server_url: str


@dataclass
class RunConfig:
    mcp_versions: list[MCPVersion]
    models: list[Literal["mistral:mistral-medium-latest"]]
    system_prompts: list[str]

    @property
    def config(self) -> pd.DataFrame:
        df_servers = pd.DataFrame(self.mcp_versions)
        df_models = pd.DataFrame({"model": self.models})
        df_prompts = pd.DataFrame({"system_prompt": self.system_prompts})
        return df_servers.merge(df_models, how="cross").merge(df_prompts, how="cross")


mcp_versions = [
    MCPVersion(**{"mcp_version": "1", "mcp_server_url": "https://mcp.data.gouv.fr/mcp"})
]
models = ["mistral:mistral-medium-latest"]
system_prompts = [
    "You are a data assistant using an MCP server. Use MCP tools when relevant to retrieve factual information. Provide a final answer to the user."
]
