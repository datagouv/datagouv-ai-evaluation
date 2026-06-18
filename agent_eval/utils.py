import asyncio
import logging

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent_eval._env import ENV_VALUES

logger = logging.getLogger(__name__)

# Waits in seconds between retries when the provider returns 429.
# First wait >= 65s so the 10 req/min window is guaranteed to reset.
_RATE_LIMIT_BACKOFF = [65, 70, 80, 90, 120]


class CompatibleOpenAIChatModel(OpenAIChatModel):
    """OpenAIChatModel with three fixes for OpenAI-compatible providers:

    1. Retries 429 rate-limit errors at the individual request level so the
       agent resumes from the failing turn rather than restarting from scratch.
    2. Patches tool_calls with type=null (e.g. Mistral) to type="function"
       before pydantic-ai re-validates the response.
    3. Flattens message content returned as a list of OpenAI "content parts"
       (e.g. Mistral: [{"type": "text", "text": "..."}, ...]) into a plain
       string, since pydantic-ai's ChatCompletion expects content: str.

    Accumulated backoff wait time is tracked in `rate_limit_wait_ms` so callers
    can compute net latency (actual inference time, excluding quota delays).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_wait_ms: float = 0.0

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
                self.rate_limit_wait_ms += wait * 1000
            try:
                return await super().request(
                    messages, model_settings, model_request_parameters
                )
            except ModelHTTPError as exc:
                if exc.status_code == 429 and attempt < len(_RATE_LIMIT_BACKOFF):
                    continue
                raise

    def _process_response(self, response):
        # Patch quirks from OpenAI-compatible providers before pydantic-ai re-validates.
        if not isinstance(response, str):
            for choice in response.choices:
                # Mistral may return content as a list of content parts
                # ([{"type": "text", "text": "..."}, ...]) instead of a plain string.
                content = getattr(choice.message, "content", None)
                if isinstance(content, list):
                    choice.message.content = "".join(
                        (
                            part.get("text", "")
                            if isinstance(part, dict)
                            else getattr(part, "text", "") or ""
                        )
                        for part in content
                    )
                # Mistral also sometimes omits the type field on tool_calls,
                # returning null instead of "function".
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
    - provider_api_key : str, name of the environment variable in .env where the API key is stored
    - provider : str, name of the provider (for logging purpose)
    """
    # TODO: create a model_config class to validate parsing for judge (judge_model.py) and task models (benchmark/loader.py)
    model_name = model_config["name"]
    provider_base_url = model_config["provider_base_url"]
    api_key = ENV_VALUES.get(model_config["provider_api_key"])
    if not api_key:
        raise ValueError(
            f"Please set the API key in environment variable {model_config['provider_api_key']} "
            f"for provider: {model_config['provider']}"
        )

    return CompatibleOpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=provider_base_url, api_key=api_key),
    )
