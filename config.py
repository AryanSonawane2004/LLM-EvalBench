# config.py

import os
from pathlib import Path


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "LLM-EvalBench"

APP_VERSION = "1.0.0"

APP_ENV = os.getenv(
    "APP_ENV",
    "development",
)


# ============================================================
# PROVIDER
# ============================================================

# LLM-EvalBench currently supports OpenRouter only.

PROVIDER = "openrouter"

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)


# ============================================================
# SERVER
# ============================================================

HOST = os.getenv(
    "HOST",
    "0.0.0.0",
)

PORT = int(
    os.getenv(
        "PORT",
        "8000",
    )
)


# ============================================================
# FRONTEND
# ============================================================

# Local development:
#   http://localhost:3000
#
# Production:
#   Set FRONTEND_ORIGIN in Render to your
#   GitHub Pages URL.

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:3000",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

DATA_DIR = (
    BASE_DIR / "data"
)

BENCHMARK_FILE = (
    DATA_DIR / "benchmark.jsonl"
)


# ============================================================
# OPENROUTER ENDPOINTS
# ============================================================

OPENROUTER_MODELS_URL = (
    f"{OPENROUTER_BASE_URL}/models"
)

OPENROUTER_CHAT_URL = (
    f"{OPENROUTER_BASE_URL}/chat/completions"
)


# ============================================================
# HTTP
# ============================================================

REQUEST_TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT",
        "120",
    )
)


# ============================================================
# LLM DEFAULTS
# ============================================================

DEFAULT_TEMPERATURE = float(
    os.getenv(
        "DEFAULT_TEMPERATURE",
        "0.0",
    )
)

DEFAULT_MAX_TOKENS = int(
    os.getenv(
        "DEFAULT_MAX_TOKENS",
        "1024",
    )
)


# ============================================================
# BENCHMARK
# ============================================================

BENCHMARK_MAX_QUESTIONS = int(
    os.getenv(
        "BENCHMARK_MAX_QUESTIONS",
        "20",
    )
)


# ============================================================
# CORS
# ============================================================

ALLOWED_ORIGINS = [
    FRONTEND_ORIGIN,
]

# Always allow local development.

if (
    "http://localhost:3000"
    not in ALLOWED_ORIGINS
):

    ALLOWED_ORIGINS.append(
        "http://localhost:3000"
    )

if (
    "http://127.0.0.1:3000"
    not in ALLOWED_ORIGINS
):

    ALLOWED_ORIGINS.append(
        "http://127.0.0.1:3000"
    )


# ============================================================
# SECURITY
# ============================================================

# API keys are supplied by users at runtime.
#
# NEVER put an OpenRouter API key in this file.
#
# Do NOT create:
#
# OPENROUTER_API_KEY = "sk-or-v1-..."
#
# The frontend sends the user's key through X-API-Key,
# and the backend forwards it to OpenRouter when required.

API_KEY_HEADER = "X-API-Key"


# ============================================================
# APPLICATION INFO
# ============================================================

ARCHITECTURE = (
    "openrouter-only"
)


def get_config() -> dict:
    """
    Return the public application configuration.

    No secrets or API keys are included.
    """

    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "environment": APP_ENV,
        "provider": PROVIDER,
        "architecture": ARCHITECTURE,
        "openrouter_base_url": (
            OPENROUTER_BASE_URL
        ),
        "frontend_origin": (
            FRONTEND_ORIGIN
        ),
        "request_timeout": (
            REQUEST_TIMEOUT
        ),
        "default_temperature": (
            DEFAULT_TEMPERATURE
        ),
        "default_max_tokens": (
            DEFAULT_MAX_TOKENS
        ),
        "benchmark_max_questions": (
            BENCHMARK_MAX_QUESTIONS
        ),
    }