import asyncio
import logging

from dotenv import load_dotenv, dotenv_values

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv(override=True)
config = dotenv_values(".env")

logger = logging.getLogger(__name__)

# Waits in seconds between retries when the provider returns 429.
# First wait >= 65s so the 10 req/min window is guaranteed to reset.
_RATE_LIMIT_BACKOFF = [65, 70, 80, 90, 120]


class CompatibleOpenAIChatModel(OpenAIChatModel):
    """OpenAIChatModel with two fixes for OpenAI-compatible providers:

    1. Retries 429 rate-limit errors at the individual request level so the
       agent resumes from the failing turn rather than restarting from scratch.
    2. Patches tool_calls with type=null (e.g. Mistral) to type="function"
       before pydantic-ai re-validates the response.
    """

    async def request(self, messages, model_settings, model_request_parameters):
        for attempt, wait in enumerate([0] + _RATE_LIMIT_BACKOFF):
            if wait:
                logger.warning(
                    "Rate limited (429), retrying in %ds (attempt %d/%d)",
                    wait,
                    attempt,
                    len(_RATE_LIMIT_BACKOFF) + 1,
                )
                await asyncio.sleep(wait)
            try:
                return await super().request(
                    messages, model_settings, model_request_parameters
                )
            except ModelHTTPError as exc:
                if exc.status_code == 429 and attempt < len(_RATE_LIMIT_BACKOFF):
                    continue
                raise

    def _process_response(self, response):
        # Some OpenAI-compatible providers (e.g. Mistral) omit the type field on
        # tool_calls, returning null instead of "function". Patch before pydantic-ai
        # re-validates the response.
        if not isinstance(response, str):
            for choice in response.choices:
                for tc in getattr(choice.message, "tool_calls", None) or []:
                    if getattr(tc, "type", None) is None:
                        tc.type = "function"
        return super()._process_response(response)


def get_model_config_object(model_config: dict) -> CompatibleOpenAIChatModel:
    """Returns an improved OpenAIChatModel object, that is compatible with various
    models and providers used in this project (MistralAI, Albert API, etc).
    Takes model_config dict that must contains :
    - name : str, model_name,
    - provider_base_url : str, api url to send request to
    - provider_token : str, name of the environment variable in .env where the token is stored
    - provider : str, name of the provider (for logging purpose)
    """
    # TODO: create a model_config class to validate parsing for judge (judge_model.py) and task models (benchmark/loader.py)
    model_name = model_config["name"]
    provider_base_url = model_config["provider_base_url"]
    provider_token = config.get(model_config["provider_token"], None)
    if not provider_token:
        raise TypeError(
            f"Please set the API token in the following environment variable {model_config['provider_token']}"
            f"for the following provider: {model_config['provider']}"
        )

    return CompatibleOpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=provider_base_url, api_key=provider_token),
    )
