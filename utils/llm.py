"""
Centralized LLM Configuration
-----------------------------
Provides a single source of truth for LLM model selection, client creation,
timeout, retry, and error handling for all agents.

Model name is read from .env (LLM_MODEL) with a configurable fallback (LLM_FALLBACK_MODEL).
"""
import os
import logging
import time
from typing import Optional

from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration from .env ---
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_FALLBACK_MODEL = "mistralai/ministral-8b-2512"
DEFAULT_TIMEOUT_MS = 120000
DEFAULT_MAX_RETRIES = 3

LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL)
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS)))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))


def get_client() -> OpenRouter:
    """Returns a configured OpenRouter client."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to initialize the LLM client.")
    return OpenRouter(api_key=api_key)


def get_model() -> str:
    """Returns the primary LLM model name from .env."""
    return LLM_MODEL


def get_fallback_model() -> str:
    """Returns the fallback LLM model name from .env."""
    return LLM_FALLBACK_MODEL


def chat_completion(
    messages: list,
    model: Optional[str] = None,
    stream: bool = False,
    timeout_ms: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> str:
    """
    Sends a chat completion request to OpenRouter with retry and fallback logic.

    Args:
        messages: List of message dicts (role, content).
        model: Model to use. Defaults to LLM_MODEL from .env.
        stream: Whether to stream the response.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Number of retry attempts on failure.

    Returns:
        The response content as a string.

    Raises:
        RuntimeError: If all retries and fallback models fail.
    """
    client = get_client()
    model = model or LLM_MODEL
    timeout_ms = timeout_ms or LLM_TIMEOUT_MS
    max_retries = max_retries or LLM_MAX_RETRIES

    models_to_try = [model]
    if model != LLM_FALLBACK_MODEL:
        models_to_try.append(LLM_FALLBACK_MODEL)

    last_error = None

    for attempt_model in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM call: model={attempt_model}, attempt={attempt + 1}/{max_retries}")
                response = client.chat.send(
                    model=attempt_model,
                    messages=messages,
                    timeout_ms=timeout_ms,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                last_error = "Empty response from LLM"
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call failed (model={attempt_model}, attempt={attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")


def chat_completion_stream(
    messages: list,
    model: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    max_retries: Optional[int] = None,
):
    """
    Sends a streaming chat completion request to OpenRouter.

    Args:
        messages: List of message dicts (role, content).
        model: Model to use. Defaults to LLM_MODEL from .env.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Number of retry attempts on failure.

    Yields:
        Chunks of response content as strings.
    """
    client = get_client()
    model = model or LLM_MODEL
    timeout_ms = timeout_ms or LLM_TIMEOUT_MS
    max_retries = max_retries or LLM_MAX_RETRIES

    models_to_try = [model]
    if model != LLM_FALLBACK_MODEL:
        models_to_try.append(LLM_FALLBACK_MODEL)

    last_error = None

    for attempt_model in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM stream call: model={attempt_model}, attempt={attempt + 1}/{max_retries}")
                stream = client.chat.send(
                    model=attempt_model,
                    messages=messages,
                    stream=True,
                    timeout_ms=timeout_ms,
                )
                for chunk in stream:
                    if chunk and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM stream call failed (model={attempt_model}, attempt={attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    raise RuntimeError(f"All LLM stream attempts failed. Last error: {last_error}")