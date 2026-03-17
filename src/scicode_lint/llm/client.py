"""LLM client for vLLM with structured output.

CRITICAL: Thinking Models Require `guided_json`, NOT `response_format`
======================================================================

Qwen3 (and other thinking models) output reasoning in `<think>...</think>` blocks
BEFORE producing the final JSON answer. This thinking phase is essential for accuracy.

The Problem
-----------
OpenAI-compatible `response_format: json_schema` forces immediate JSON output,
SKIPPING the thinking phase entirely:

    # guided_json (correct) - model thinks first, then produces JSON
    <think>
    Let me analyze this code. The model.eval() is called on line 42,
    then training happens on lines 54-60 without model.train()...
    </think>
    {"detected": true, "reasoning": "Training after eval without train mode"}

    # response_format json_schema (WRONG) - no thinking, immediate JSON
    {"detected": true, "reasoning": "Training after eval"}

Impact: Using `json_schema` instead of `guided_json` drops accuracy from ~99% to ~78%.

Root Cause
----------
- `guided_json` (in extra_body): Uses vLLM's XGrammar/Outlines backend to constrain
  output AFTER the model completes its thinking phase
- `response_format: json_schema`: Activates OpenAI compatibility mode that expects
  immediate JSON output, suppressing the `<think>` blocks

This is specific to models with VISIBLE thinking tokens (like Qwen3's <think> blocks).
OpenAI's o-series reasoning models use INTERNAL reasoning tokens (hidden from output),
so json_schema works fine for them. But Qwen3's visible thinking gets suppressed.

vLLM's guided decoding (XGrammar/Outlines) is enabled by default - no special config needed.

Correct Usage
-------------
    # CORRECT - preserves thinking
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"guided_json": json_schema},
    )

    # WRONG - skips thinking phase, ~20% accuracy drop
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {...}},
    )
"""

import json
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

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


class MissingLocationError(ValueError):
    """Raised when LLM returns detected='yes' but no location.

    This is a specific validation error that indicates the model understood
    there was an issue but failed to identify where it occurs in the code.
    The error message includes details for debugging and logging.
    """

    def __init__(self, detected: str, reasoning: str, confidence: float = 0.0):
        self.detected = detected
        self.reasoning = reasoning
        self.confidence = confidence
        super().__init__(
            f"LLM returned detected='{detected}' but no location. Reasoning: {reasoning[:100]}..."
        )


# Correction prompt added when retrying after missing location
# Includes previous response to help model understand the issue
MISSING_LOCATION_CORRECTION = """

CORRECTION REQUIRED - Your previous response was INVALID:
- You said: detected="{detected}"
- Your reasoning: "{reasoning}"
- But you provided: location=null

This is invalid. If you detect an issue, you MUST provide the location.
Identify the function, class, or method name where the issue occurs.

OPTIONS:
1. If you CAN identify the location: provide it as:
   "location": {{"name": "function_name", "location_type": "function", "near_line": 15}}
2. If you CANNOT identify a specific location: change your answer to detected="no"
   (An issue without identifiable location is not actionable)

Respond with valid JSON including proper location OR change to detected="no".
"""


class LLMClient(ABC):
    """Abstract base class for LLM clients with structured output."""

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
        self._max_model_len: int = 0

        # Use OpenAI SDK for vLLM's OpenAI-compatible API
        try:
            from openai import AsyncOpenAI
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

        self._max_model_len = self.config.max_model_len

    @staticmethod
    def _extract_thinking(text: str) -> tuple[str, str | None]:
        """
        Extract and remove <think>...</think> tags from response.

        Qwen3 models in thinking mode output reasoning in <think> tags before the JSON.
        Handles both closed tags and truncated/unclosed tags (when output is cut off).

        Returns:
            Tuple of (cleaned_text, thinking_content).
            thinking_content is None if no thinking tags were found.
        """
        import re

        thinking_content = None

        # Try to find closed thinking tags first
        thinking_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            # Remove the closed thinking block
            cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        elif text.strip().startswith("<think>"):
            # Handle unclosed/truncated thinking tags (output was cut off)
            # Everything after <think> is thinking content, no JSON available
            thinking_content = text[7:].strip()  # Skip "<think>"
            cleaned_text = ""  # No JSON content available
            logger.warning(
                f"Truncated thinking detected ({len(thinking_content)} chars). "
                "Model may have exhausted output tokens on reasoning."
            )
        else:
            cleaned_text = text

        return cleaned_text.strip(), thinking_content

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
    def _parse_and_validate(response_text: str, schema: type[T]) -> T:
        """
        Parse JSON response and validate against Pydantic schema.

        Strips thinking tags (Qwen3) and markdown code fences that models often add.
        If the schema has a 'thinking' field, the extracted thinking content will be
        stored there.

        Args:
            response_text: Raw JSON string from LLM (may have thinking tags or markdown fences)
            schema: Pydantic model class to validate against

        Returns:
            Validated Pydantic model instance

        Raises:
            json.JSONDecodeError: If JSON parsing fails
            MissingLocationError: If detected='yes'/'context-dependent' but location is None
            ValidationError: If other schema validation fails
        """
        # Extract thinking content (Qwen3) and strip markdown code fences
        cleaned_text, thinking_content = VLLMClient._extract_thinking(response_text)
        cleaned_text = VLLMClient._strip_markdown_fences(cleaned_text)

        if thinking_content:
            logger.debug(f"Extracted thinking content ({len(thinking_content)} chars)")

        try:
            response_data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Original response: {response_text[:500]}")
            logger.error(f"Cleaned response: {cleaned_text[:500]}")
            raise

        # If schema has 'thinking' field, add extracted thinking content
        if thinking_content and "thinking" in schema.model_fields:
            response_data["thinking"] = thinking_content

        # Check for missing location before full validation (to raise specific error)
        detected = response_data.get("detected")
        location = response_data.get("location")
        if detected in ("yes", "context-dependent") and not location:
            reasoning = response_data.get("reasoning", "")
            confidence = response_data.get("confidence", 0.0)
            raise MissingLocationError(
                detected=detected, reasoning=reasoning, confidence=confidence
            )

        try:
            result = schema.model_validate(response_data)
        except ValidationError as e:
            logger.error(f"Failed to validate response against schema {schema.__name__}: {e}")
            logger.error(f"Response data: {response_data}")
            raise

        return result

    def _handle_response(
        self,
        response_text: str,
        schema: type[T],
        attempt: int,
        max_attempts: int,
        start_time: float,
        label: str,
    ) -> tuple[T | None, str | None]:
        """Handle LLM response: parse, validate, and manage retry logic.

        Args:
            response_text: Raw LLM response text
            schema: Pydantic model class to validate against
            attempt: Current attempt number (0-indexed)
            max_attempts: Maximum number of attempts
            start_time: Time when the call started (for elapsed logging)
            label: Log label ("vLLM call" or "Async vLLM call")

        Returns:
            Tuple of (result, correction_prompt). If result is not None, the call
            succeeded. If result is None, correction_prompt contains the retry prompt.

        Raises:
            ValueError: If JSON parsing or schema validation fails (non-retryable)
        """
        try:
            result = self._parse_and_validate(response_text, schema)
            elapsed = time.time() - start_time
            logger.info(f"{label} completed in {elapsed:.2f}s for {schema.__name__}")
            return result, None

        except MissingLocationError as e:
            if attempt < max_attempts - 1:
                correction = MISSING_LOCATION_CORRECTION.format(
                    detected=e.detected, reasoning=e.reasoning[:200]
                )
                logger.warning(
                    f"Missing location (detected='{e.detected}'), retrying with correction"
                )
                return None, correction
            else:
                # Final attempt failed - flip to "no" since we can't identify location
                logger.warning(
                    f"Missing location after retry - flipping to 'no': {e.reasoning[:100]}"
                )
                flipped_data = {
                    "detected": "no",
                    "location": None,
                    "confidence": 0.5,
                    "reasoning": (
                        f"Originally detected='{e.detected}' but could not identify "
                        f"specific location. Flipped to 'no' since unlocatable "
                        f"issues are not actionable. Original reasoning: {e.reasoning[:150]}"
                    ),
                }
                return schema.model_validate(flipped_data), None

        except (json.JSONDecodeError, ValidationError) as e:
            error_type = (
                "JSON parse" if isinstance(e, json.JSONDecodeError) else "schema validation"
            )
            logger.error(f"LLM response {error_type} failed")
            raise ValueError(
                f"LLM returned invalid response ({error_type} error). "
                f"Model not producing valid structured output. Error: {e}"
            ) from e

    def _build_api_params(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build parameters for the OpenAI API call."""
        return {
            "model": self.config.model_served_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "extra_body": {"guided_json": json_schema, "top_k": self.config.top_k},
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": max_tokens,
        }

    async def async_complete_structured(
        self, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        """
        Get structured completion from vLLM asynchronously using OpenAI SDK.

        Uses vLLM's guided_json for schema-constrained output that preserves thinking.
        Requires vLLM with guided decoding support (XGrammar/Outlines backend).

        If the model returns detected='yes' but missing location, retries once with a
        correction prompt to enforce location extraction.

        Concurrent requests with shared prefixes benefit from automatic prefix caching.
        Uses OpenAI AsyncClient for true async concurrency.
        """
        start_time = time.time()
        logger.debug(f"Starting async vLLM call for {schema.__name__}")

        json_schema = schema.model_json_schema()
        max_tokens = self.config.max_completion_tokens or 2048
        correction_prompt: str | None = None
        max_attempts = 2

        for attempt in range(max_attempts):
            current_prompt = user_prompt
            if correction_prompt:
                current_prompt = user_prompt + correction_prompt
                logger.info(f"Retrying with correction prompt (attempt {attempt + 1})")

            # IMPORTANT: Use guided_json in extra_body, NOT response_format with json_schema.
            # Qwen3 outputs <think>...</think> blocks BEFORE the JSON.
            # Using json_schema drops accuracy from ~99% to ~78% because it skips thinking.
            try:
                completion = await self._async_client.chat.completions.create(
                    **self._build_api_params(system_prompt, current_prompt, json_schema, max_tokens)
                )
            except Exception as e:
                raise RuntimeError(
                    f"vLLM structured output failed: {e}\n"
                    "guided_json with XGrammar/Outlines is required for thinking models."
                ) from e

            response_text = completion.choices[0].message.content
            if response_text is None:
                raise ValueError("LLM returned empty response")

            result, correction_prompt = self._handle_response(
                response_text, schema, attempt, max_attempts, start_time, "Async vLLM call"
            )
            if result is not None:
                return result

        raise RuntimeError("Unexpected: all attempts exhausted without result or exception")

    def get_max_model_len(self) -> int:
        """Get maximum context length in tokens.

        Returns:
            Maximum context length supported by the model

        Example:
            >>> client = VLLMClient(config)
            >>> max_len = client.get_max_model_len()
            >>> print(f"Max tokens: {max_len}")
        """
        return self._max_model_len


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
        "  2. Start: vllm serve RedHatAI/Qwen3-8B-FP8-dynamic\n"
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
    # Auto-detect base_url if not specified
    if not config.base_url:
        detected_url, _ = detect_vllm()
        # Create a copy to avoid mutating the caller's config
        config = config.model_copy(update={"base_url": detected_url})

    return VLLMClient(config)
