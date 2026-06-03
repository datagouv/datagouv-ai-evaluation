import logging
from pathlib import Path

import yaml
from pydantic_ai.providers.openai import OpenAIProvider

from agent_eval._env import ENV_VALUES
from agent_eval.utils import CompatibleOpenAIChatModel

logger = logging.getLogger(__name__)


class JudgeModel(CompatibleOpenAIChatModel):
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        logger.info("Judge model: %s", raw.get("name"))

        model_name = raw["name"]
        provider_base_url = raw["provider_base_url"]
        provider_token = ENV_VALUES.get(raw["provider_token"])
        if not provider_token:
            raise ValueError(
                f"Please set the API token in environment variable {raw['provider_token']} "
                f"for provider: {raw['provider']}"
            )

        super().__init__(
            model_name,
            provider=OpenAIProvider(base_url=provider_base_url, api_key=provider_token),
        )
