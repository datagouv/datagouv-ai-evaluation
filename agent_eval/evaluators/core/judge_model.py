import logging
from pathlib import Path

import yaml

from agent_eval._env import ENV_VALUES
from agent_eval.utils import CompatibleOpenAIChatModel, make_openai_provider

logger = logging.getLogger(__name__)


class JudgeModel(CompatibleOpenAIChatModel):
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        logger.info("Judge model: %s", raw.get("name"))

        model_name = raw["name"]
        provider_base_url = raw["provider_base_url"]
        api_key = ENV_VALUES.get(raw["provider_api_key"])
        if not api_key:
            raise ValueError(
                f"Please set the API key in environment variable {raw['provider_api_key']} "
                f"for provider: {raw['provider']}"
            )

        super().__init__(
            model_name,
            provider=make_openai_provider(provider_base_url, api_key),
        )
