# models/model_runner.py

import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    OPENROUTER_CHAT_URL,
    OPENROUTER_MODELS_URL,
    REQUEST_TIMEOUT,
)


# ============================================================
# CONFIG
# ============================================================

PROVIDER = "openrouter"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0

# Small delay between successful OpenRouter requests.
# This is especially useful for free models.
REQUEST_DELAY_SECONDS = 1.0


# ============================================================
# HEADERS
# ============================================================

def build_headers(api_key: str) -> Dict[str, str]:
    if not api_key or not api_key.strip():
        raise PermissionError(
            "OpenRouter API key is missing."
        )

    return {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# ERROR HELPERS
# ============================================================

def _extract_error_message(response: requests.Response) -> str:
    try:
        data = response.json()

        if isinstance(data, dict):
            error = data.get("error")

            if isinstance(error, dict):
                message = error.get("message")

                if message:
                    return str(message)

            if isinstance(error, str):
                return error

            detail = data.get("detail")

            if detail:
                return str(detail)

    except Exception:
        pass

    text = response.text.strip()

    if text:
        return text[:1000]

    return f"HTTP {response.status_code}"

def _is_non_retryable_rate_limit(message: str) -> bool:
    """Return True when OpenRouter indicates a quota that won't recover by waiting."""
    if not message:
        return False

    message_lower = message.lower()

    non_retryable_markers = [
        "free-models-per-day",
        "daily limit",
        "daily quota",
        "quota exceeded",
    ]

    return any(marker in message_lower for marker in non_retryable_markers)

def _is_rate_limit(response: requests.Response) -> bool:
    return response.status_code == 429


def _retry_delay(
    response: requests.Response,
    attempt: int,
) -> float:
    """
    Determine retry delay.

    Prefer Retry-After when OpenRouter provides it.
    Otherwise use exponential backoff.
    """

    retry_after = response.headers.get(
        "Retry-After"
    )

    if retry_after:
        try:
            delay = float(retry_after)

            return min(
                max(delay, 0.0),
                MAX_BACKOFF_SECONDS,
            )

        except ValueError:
            pass

    delay = INITIAL_BACKOFF_SECONDS * (
        2 ** attempt
    )

    return min(
        delay,
        MAX_BACKOFF_SECONDS,
    )


def _raise_openrouter_error(
    response: requests.Response,
) -> None:

    message = _extract_error_message(
        response
    )

    status = response.status_code

    if status in (401, 403):
        raise PermissionError(
            "OpenRouter API key is invalid, "
            "expired, or does not have permission "
            "to access this resource."
        )

    if status == 402:
        raise RuntimeError(
            "Insufficient OpenRouter credits. "
            "The selected model requires credits "
            "or the account has insufficient balance."
        )

    if status == 429:
        raise RuntimeError(
            f"OpenRouter rate limit exceeded: "
            f"{message}"
        )

    if status == 408:
        raise TimeoutError(
            f"OpenRouter request timed out: "
            f"{message}"
        )

    if 500 <= status < 600:
        raise RuntimeError(
            f"OpenRouter server error "
            f"({status}): {message}"
        )

    raise RuntimeError(
        f"OpenRouter API error "
        f"({status}): {message}"
    )


# ============================================================
# RESPONSE HANDLING
# ============================================================

def _handle_response(
    response: requests.Response,
) -> Dict[str, Any]:

    if not response.ok:
        _raise_openrouter_error(
            response
        )

    try:
        return response.json()

    except ValueError as exc:
        raise RuntimeError(
            "OpenRouter returned invalid JSON."
        ) from exc


# ============================================================
# MODEL NORMALIZATION
# ============================================================

def _normalize_model(
    model: Dict[str, Any],
) -> Dict[str, Any]:

    model_id = (
        model.get("id")
        or model.get("model")
        or ""
    )

    architecture = (
        model.get("architecture")
        or {}
    )

    input_modalities = (
        architecture.get(
            "input_modalities",
            [],
        )
        or []
    )

    output_modalities = (
        architecture.get(
            "output_modalities",
            [],
        )
        or []
    )

    supported_parameters = (
        model.get(
            "supported_parameters",
            [],
        )
        or []
    )

    benchmark_compatible = (
        "text" in input_modalities
        and "text" in output_modalities
    )

    pricing = model.get(
        "pricing",
        {},
    ) or {}

    return {
        "id": model_id,
        "name": (
            model.get("name")
            or model_id
        ),
        "provider": PROVIDER,
        "capabilities": (
            ["text_generation"]
            if benchmark_compatible
            else []
        ),
        "benchmark_compatible": (
            benchmark_compatible
        ),
        "input_modalities": (
            input_modalities
        ),
        "output_modalities": (
            output_modalities
        ),
        "supported_parameters": (
            supported_parameters
        ),
        "context_length": (
            model.get("context_length")
        ),
        "max_completion_tokens": (
            model.get(
                "max_completion_tokens"
            )
        ),
        "pricing": {
            "prompt": pricing.get(
                "prompt"
            ),
            "completion": pricing.get(
                "completion"
            ),
        },
    }


# ============================================================
# LIST MODELS
# ============================================================

def list_models(
    api_key: str,
) -> List[Dict[str, Any]]:

    response = requests.get(
        OPENROUTER_MODELS_URL,
        headers=build_headers(
            api_key
        ),
        timeout=REQUEST_TIMEOUT,
    )

    data = _handle_response(
        response
    )

    raw_models = data.get(
        "data",
        [],
    )

    models = [
        _normalize_model(model)
        for model in raw_models
    ]

    # Benchmark-compatible models first.
    models.sort(
        key=lambda item: (
            not item[
                "benchmark_compatible"
            ],
            item["name"].lower(),
        )
    )

    return models


# ============================================================
# VERIFY API KEY
# ============================================================

def verify_api_key(
    api_key: str,
) -> Dict[str, Any]:

    try:
        models = list_models(
            api_key
        )

        benchmark_models = [
            model
            for model in models
            if model.get(
                "benchmark_compatible",
                False,
            )
        ]

        return {
            "verified": True,
            "provider": PROVIDER,
            "model_count": len(
                models
            ),
            "benchmark_model_count": len(
                benchmark_models
            ),
        }

    except PermissionError:
        return {
            "verified": False,
            "provider": PROVIDER,
            "error": (
                "Invalid or unauthorized "
                "OpenRouter API key."
            ),
        }

    except Exception as exc:
        return {
            "verified": False,
            "provider": PROVIDER,
            "error": str(exc),
        }


# ============================================================
# GET MODEL
# ============================================================

def get_model(
    model: str,
    api_key: str,
) -> Dict[str, Any]:

    models = list_models(
        api_key
    )

    target = model.strip().lower()

    for item in models:

        if item["id"].lower() == target:
            return item

    raise ValueError(
        f"OpenRouter model '{model}' "
        f"was not found in the current catalog."
    )


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate_model(
    model: str,
    api_key: str,
    require_benchmark_compatible: bool = True,
) -> Dict[str, Any]:

    model_info = get_model(
        model,
        api_key,
    )

    if (
        require_benchmark_compatible
        and not model_info.get(
            "benchmark_compatible",
            False,
        )
    ):
        raise ValueError(
            f"Model '{model}' is not compatible "
            f"with text-to-text benchmarking."
        )

    return model_info


# ============================================================
# MESSAGES
# ============================================================

def _build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    return messages


# ============================================================
# OPENROUTER CHAT REQUEST
# ============================================================

def _chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format is not None:
        payload["response_format"] = response_format

    last_error = None

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers=build_headers(
                    api_key
                ),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.ok:

                if (
                    REQUEST_DELAY_SECONDS
                    > 0
                ):
                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

                return _handle_response(
                    response
                )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if _is_rate_limit(response):

                last_error = _extract_error_message(
                    response
                )

                # Permanent quota/rate-limit errors should not be retried.
                if _is_non_retryable_rate_limit(
                    last_error
                ):
                    raise RuntimeError(
                        "OpenRouter rate limit exceeded: "
                        f"{last_error}"
                    )

                # Temporary rate limit: retry with exponential backoff.
                if attempt >= MAX_RETRIES:
                    raise RuntimeError(
                        "OpenRouter rate limit "
                        f"persisted after "
                        f"{MAX_RETRIES} retries: "
                        f"{last_error}"
                    )

                delay = _retry_delay(
                    response,
                    attempt,
                )

                print(
                    "[OpenRouter] "
                    f"Rate limited. "
                    f"Retry {attempt + 1}/"
                    f"{MAX_RETRIES} "
                    f"in {delay:.1f}s."
                )

                time.sleep(delay)
                continue

            # ------------------------------------------------
            # NON-RETRYABLE ERROR
            # ------------------------------------------------

            _raise_openrouter_error(
                response
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:

            last_error = str(exc)

            if attempt >= MAX_RETRIES:
                raise TimeoutError(
                    "OpenRouter request failed "
                    f"after {MAX_RETRIES} retries: "
                    f"{exc}"
                ) from exc

            delay = min(
                INITIAL_BACKOFF_SECONDS
                * (2 ** attempt),
                MAX_BACKOFF_SECONDS,
            )

            print(
                "[OpenRouter] "
                f"Network/timeout error. "
                f"Retry {attempt + 1}/"
                f"{MAX_RETRIES} "
                f"in {delay:.1f}s."
            )

            time.sleep(
                delay
            )

    raise RuntimeError(
        "OpenRouter request failed: "
        f"{last_error}"
    )


# ============================================================
# RUN MODEL
# ============================================================

def run_model(
    model: str,
    prompt: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: Optional[Dict[str, Any]] = None,
):

    # Validate model before generation.
    validate_model(
        model=model,
        api_key=api_key,
        require_benchmark_compatible=True,
    )

    messages = _build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    data = _chat_completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )

    choices = data.get(
        "choices",
        [],
    )

    if not choices:
        raise RuntimeError(
            "OpenRouter returned no choices."
        )

    message = (
        choices[0].get(
            "message",
            {},
        )
        or {}
    )

    content = message.get(
        "content"
    )

    if content is None:
        raise RuntimeError(
            "OpenRouter returned an empty "
            "response content."
        )

    if not isinstance(
        content,
        str,
    ):
        content = str(content)

    content = content.strip()

    if not content:
        raise RuntimeError(
            "OpenRouter returned an empty "
            "response."
        )

    return content


# ============================================================
# DETAILED RUN
# ============================================================

def run_model_detailed(
    model: str,
    prompt: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:

    validate_model(
        model=model,
        api_key=api_key,
        require_benchmark_compatible=True,
    )

    messages = _build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    data = _chat_completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choices = data.get(
        "choices",
        [],
    )

    if not choices:
        raise RuntimeError(
            "OpenRouter returned no choices."
        )

    choice = choices[0]

    message = (
        choice.get(
            "message",
            {},
        )
        or {}
    )

    content = message.get(
        "content"
    )

    if content is None:
        raise RuntimeError(
            "OpenRouter returned empty content."
        )

    if not isinstance(
        content,
        str,
    ):
        content = str(content)

    content = content.strip()

    if not content:
        raise RuntimeError(
            "OpenRouter returned empty content."
        )

    return {
        "content": content,
        "usage": data.get(
            "usage"
        ),
        "id": data.get(
            "id"
        ),
        "finish_reason": choice.get(
            "finish_reason"
        ),
        "model": data.get(
            "model",
            model,
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import os

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY first."
        )

    result = run_model(
        model="qwen/qwen3-8b",
        prompt="What is 2 + 2?",
        api_key=api_key,
    )

    print(result)