"""LLM client for vLLM with structured output."""

import json
import time
import warnings
from abc import ABC, abstractmethod
from typing import TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scicode_lint.config import LLMConfig

# Suppress Pydantic serialization warning for OpenAI SDK's ParsedChatCompletionMessage.parsed field
# Root cause: When using OpenAI SDK's structured outputs with LangChain, the parsed field
# (TypeVar defaulting to None) receives our Pydantic model, triggering serialization warnings.
# This is a known upstream issue in openai-python v2.21.0+
# Reference: https://github.com/openai/openai-python/issues/2872
# The warning is harmless - structured output parsing works correctly despite the warning.
# This targeted suppression only affects the specific pydantic.main module warning.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main"
)

T = TypeVar("T", bound=BaseModel)

# Note: This minimal system prompt is kept for backward compatibility
# but is no longer used by async_complete_structured (which now respects the provided system_prompt)
VLLM_SYSTEM_PROMPT = """You are a code analyzer. Analyze the code and answer the detection question.
Be conservative - only report bugs you're confident about (confidence >= 0.7).
Use function/class/method names for locations, not line numbers."""


class LLMClient(ABC):
    """Abstract base class for LLM clients with structured output."""

    @abstractmethod
    def complete_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        """
        Get structured completion from LLM.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Pydantic model class for response structure

        Returns:
            Validated Pydantic model instance
        """
        pass

    @abstractmethod
    async def async_complete_structured(
        self, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        """
        Get structured completion from LLM asynchronously.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Pydantic model class for response structure

        Returns:
            Validated Pydantic model instance
        """
        pass

    @abstractmethod
    def get_max_model_len(self) -> int:
        """
        Get maximum context length in tokens.

        Returns:
            Maximum context length supported by the model
        """
        pass


class VLLMClient(LLMClient):
    """
    Client for vLLM servers with structured output support.

    Supports local and remote vLLM servers only.
    Does NOT support commercial APIs (OpenAI, Anthropic, etc.) to avoid accidental costs.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize vLLM client.

        Note: Only works with vLLM servers (local or remote).
        Uses OpenAI-compatible API format but is NOT for commercial APIs.
        """
        self.config = config
        self._max_model_len: int | None = None

        # Use OpenAI SDK for vLLM's OpenAI-compatible API
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK is required for vLLM client. Install with: pip install openai"
            ) from e

        # Create async client for concurrent requests
        self._async_client = AsyncOpenAI(
            base_url=self.config.base_url + "/v1",
            api_key="dummy",  # vLLM doesn't require API keys
            timeout=float(self.config.timeout),
        )

        # Create sync client for non-async requests
        self._sync_client = OpenAI(
            base_url=self.config.base_url + "/v1",
            api_key="dummy",  # vLLM doesn't require API keys
            timeout=float(self.config.timeout),
        )

        # Auto-detect max_model_len if not specified
        if self.config.max_model_len:
            self._max_model_len = self.config.max_model_len
        else:
            self._auto_detect_max_model_len()

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Strip markdown code fences from response.

        Models often wrap JSON in ```json ... ``` despite being told not to.
        """
        text = text.strip()

        # Remove ```json ... ``` fences
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        elif text.startswith("```"):
            text = text[3:]  # Remove ```

        if text.endswith("```"):
            text = text[:-3]  # Remove trailing ```

        return text.strip()

    @staticmethod
    @retry(
        retry=retry_if_exception_type((json.JSONDecodeError, ValidationError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _parse_and_validate(response_text: str, schema: type[T]) -> T:
        """
        Parse JSON response and validate against Pydantic schema with retry.

        Retries once on JSON parse or validation errors to handle transient issues.
        Strips markdown code fences that models often add despite instructions.

        Args:
            response_text: Raw JSON string from LLM (may have markdown fences)
            schema: Pydantic model class to validate against

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If parsing or validation fails after retries
        """
        # Strip markdown code fences
        cleaned_text = VLLMClient._strip_markdown_fences(response_text)

        try:
            response_data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Original response: {response_text[:500]}")
            logger.error(f"Cleaned response: {cleaned_text[:500]}")
            raise

        try:
            result = schema.model_validate(response_data)
        except ValidationError as e:
            logger.error(f"Failed to validate response against schema {schema.__name__}: {e}")
            logger.error(f"Response data: {response_data}")
            raise

        return result

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        """
        Get structured completion from vLLM using OpenAI SDK.

        Uses vLLM's guided_json feature for enforced JSON schema compliance.
        Falls back to json_object mode if guided decoding is not supported.
        """
        start_time = time.time()
        logger.debug(f"Starting vLLM call for {schema.__name__}")

        # Get JSON schema from Pydantic model
        json_schema = schema.model_json_schema()

        # Try vLLM's guided_json first (enforces schema via constrained decoding)
        try:
            completion = self._sync_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"guided_json": json_schema},
                temperature=self.config.temperature,
            )
        except Exception as e:
            # Fallback to json_object mode if guided_json not supported
            logger.warning(f"guided_json not supported, falling back to json_object: {e}")
            completion = self._sync_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
            )

        elapsed = time.time() - start_time
        logger.info(f"vLLM call completed in {elapsed:.2f}s for {schema.__name__}")

        # Parse and validate JSON with Pydantic (with retry)
        response_text = completion.choices[0].message.content
        if response_text is None:
            raise ValueError("LLM returned empty response")

        try:
            result = self._parse_and_validate(response_text, schema)
        except (json.JSONDecodeError, ValidationError) as e:
            # After retries failed, raise clear error
            error_type = (
                "JSON parse" if isinstance(e, json.JSONDecodeError) else "schema validation"
            )
            logger.error(f"LLM response {error_type} failed after retries")
            raise ValueError(
                f"LLM returned invalid response ({error_type} error). "
                f"This indicates the model is not producing valid structured output. "
                f"Original error: {e}"
            ) from e

        return result

    async def async_complete_structured(
        self, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        """
        Get structured completion from vLLM asynchronously using OpenAI SDK.

        Uses vLLM's guided_json feature for enforced JSON schema compliance.
        Falls back to json_object mode if guided decoding is not supported.

        Concurrent requests with shared prefixes benefit from automatic prefix caching.
        Uses OpenAI AsyncClient for true async concurrency.

        vLLM is designed for concurrent requests with continuous batching and KV cache sharing.
        Parallel execution is recommended for optimal throughput.
        """
        start_time = time.time()
        logger.debug(f"Starting async vLLM call for {schema.__name__}")

        # Get JSON schema from Pydantic model
        json_schema = schema.model_json_schema()

        # Try vLLM's guided_json first (enforces schema via constrained decoding)
        # This uses XGrammar or Outlines backend to guarantee valid JSON
        try:
            completion = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"guided_json": json_schema},
                temperature=self.config.temperature,
            )
        except Exception as e:
            # Fallback to json_object mode if guided_json not supported
            logger.warning(f"guided_json not supported, falling back to json_object: {e}")
            completion = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
            )

        elapsed = time.time() - start_time
        logger.info(f"Async vLLM call completed in {elapsed:.2f}s for {schema.__name__}")

        # Parse and validate JSON with Pydantic (with retry)
        response_text = completion.choices[0].message.content
        if response_text is None:
            raise ValueError("LLM returned empty response")

        try:
            result = self._parse_and_validate(response_text, schema)
        except (json.JSONDecodeError, ValidationError) as e:
            # After retries failed, raise clear error
            error_type = (
                "JSON parse" if isinstance(e, json.JSONDecodeError) else "schema validation"
            )
            logger.error(f"LLM response {error_type} failed after retries")
            raise ValueError(
                f"LLM returned invalid response ({error_type} error). "
                f"This indicates the model is not producing valid structured output. "
                f"Original error: {e}"
            ) from e

        return result

    def _auto_detect_max_model_len(self) -> None:
        """Auto-detect max_model_len from vLLM server.

        Queries /v1/models endpoint to get model configuration.
        Falls back to conservative default (6000 tokens) if detection fails.
        """
        try:
            response = httpx.get(
                f"{self.config.base_url}/v1/models",
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_info = data["data"][0]
                    # vLLM may include max_model_len in model metadata
                    # Try common field names
                    for field in ["max_model_len", "max_length", "max_position_embeddings"]:
                        if field in model_info:
                            self._max_model_len = int(model_info[field])
                            logger.info(
                                f"Auto-detected max_model_len: {self._max_model_len} tokens"
                            )
                            return

            # Fallback: default for gemma-3-12b
            self._max_model_len = 16000
            logger.warning(
                "Could not auto-detect max_model_len from server, using default: "
                f"{self._max_model_len} tokens"
            )

        except Exception as e:
            # Fallback on any error
            self._max_model_len = 16000
            logger.warning(
                f"Error auto-detecting max_model_len ({e}), using default: "
                f"{self._max_model_len} tokens"
            )

    def get_max_model_len(self) -> int:
        """Get maximum context length in tokens.

        Returns:
            Maximum context length supported by the model

        Example:
            >>> client = VLLMClient(config)
            >>> max_len = client.get_max_model_len()
            >>> print(f"Max tokens: {max_len}")
        """
        if self._max_model_len is None:
            self._auto_detect_max_model_len()
        return self._max_model_len or 16000


def detect_vllm() -> tuple[str, str | None]:
    """
    Auto-detect vLLM server and model.

    Tries vLLM on common ports (5001, 8000).

    Returns:
        Tuple of (base_url, model_name)
        model_name is None if it couldn't be detected

    Raises:
        RuntimeError: If no vLLM server is available
    """
    # Try vLLM on common ports
    vllm_urls = ["http://localhost:5001", "http://localhost:8000"]
    for url in vllm_urls:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    # Try to get the first available model
                    model_name = None
                    if "data" in data and len(data["data"]) > 0:
                        model_name = data["data"][0].get("id")
                    logger.info(f"Detected vLLM at {url} with model: {model_name}")
                    return (url, model_name)
        except (httpx.ConnectError, httpx.TimeoutException):
            continue

    raise RuntimeError(
        "No vLLM server detected. Please start vLLM:\n\n"
        "  1. Install: pip install scicode-lint[vllm-server]\n"
        "  2. Start: vllm serve --model RedHatAI/gemma-3-12b-it-FP8-dynamic\n"
        "  3. vLLM will run on http://localhost:5001 or http://localhost:8000\n\n"
        "Note: vLLM supports CPU mode with --device cpu"
    )


def create_client(config: LLMConfig) -> LLMClient:
    """
    Create vLLM client.

    Auto-detects vLLM server if base_url is not specified.

    Args:
        config: LLM configuration

    Returns:
        vLLM client instance
    """
    # Auto-detect if no base_url specified
    if not config.base_url:
        detected_url, detected_model = detect_vllm()
        config.base_url = detected_url
        # Use detected model if user didn't specify one
        if detected_model and not config.model:
            config.model = detected_model

    return VLLMClient(config)
