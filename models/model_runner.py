import ollama
from typing import Optional, Dict, Any


DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_PREDICT = 1024


# ============================================================
# OLLAMA MODEL HELPERS
# ============================================================

def get_local_models():
    """
    Return the exact model names installed in Ollama.
    Example:
        ["qwen3:4b", "gemma:7b"]
    """

    try:
        result = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to Ollama: {exc}"
        ) from exc

    models = []

    for model in result.get("models", []):
        name = model.get("model") or model.get("name")

        if name:
            models.append(name)

    return models


def resolve_model_name(model_name: str) -> str:
    """
    Resolve a user/catalog model name to the exact model
    name registered in the local Ollama installation.

    Examples:

        qwen3
            -> qwen3:4b

        qwen3:4b
            -> qwen3:4b

        gemma:7b
            -> gemma:7b
    """

    if not model_name:
        raise ValueError("model_name cannot be empty.")

    requested = model_name.strip().lower()

    installed_models = get_local_models()

    # --------------------------------------------------------
    # 1. Exact match
    # --------------------------------------------------------

    for installed in installed_models:

        if installed.lower() == requested:
            return installed

    # --------------------------------------------------------
    # 2. Base-name match
    # --------------------------------------------------------

    requested_base = requested.split(":", 1)[0]

    candidates = []

    for installed in installed_models:

        installed_lower = installed.lower()
        installed_base = installed_lower.split(":", 1)[0]

        if installed_base == requested_base:
            candidates.append(installed)

    if candidates:

        # Prefer :latest if available
        for candidate in candidates:
            if candidate.lower() == f"{requested_base}:latest":
                return candidate

        # Otherwise use the first installed variant
        return candidates[0]

    # --------------------------------------------------------
    # 3. Not installed
    # --------------------------------------------------------

    return model_name


def is_model_installed(model_name: str) -> bool:
    """
    Check whether the requested model or one of its exact
    installed variants exists locally.
    """

    resolved = resolve_model_name(model_name)

    installed_models = get_local_models()

    return any(
        installed.lower() == resolved.lower()
        for installed in installed_models
    )


def ensure_model_available(model_name: str) -> str:
    """
    Ensure the model exists locally.

    Returns the exact Ollama model name that should be used.
    """

    if not model_name:
        raise ValueError("model_name cannot be empty.")

    resolved = resolve_model_name(model_name)

    # Already installed
    if resolved != model_name or is_model_installed(model_name):

        print(
            f"[Model Runner] {model_name} resolved to "
            f"{resolved}."
        )

        return resolved

    # --------------------------------------------------------
    # Model is not installed
    # --------------------------------------------------------

    print(
        f"[Model Runner] {model_name} is not installed."
    )

    print(
        f"[Model Runner] Pulling {model_name}..."
    )

    try:
        ollama.pull(model_name)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to pull model '{model_name}': {exc}"
        ) from exc

    print(
        f"[Model Runner] Successfully pulled "
        f"{model_name}."
    )

    # Resolve again after pull
    resolved = resolve_model_name(model_name)

    return resolved


# ============================================================
# MODEL EXECUTION
# ============================================================

def run_model(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> str:
    """
    Run a prompt against an Ollama model.

    The requested model name is first resolved to the exact
    model name installed in Ollama.
    """

    if not model:
        raise ValueError("model cannot be empty.")

    if not prompt:
        raise ValueError("prompt cannot be empty.")

    # Resolve exact installed model
    resolved_model = ensure_model_available(model)

    print(
        f"[Model Runner] Running model: "
        f"{resolved_model}"
    )

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

    try:

        response = ollama.chat(
            model=resolved_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": num_predict,
            },
        )

    except Exception as exc:

        raise RuntimeError(
            f"Ollama generation failed for "
            f"'{resolved_model}': {exc}"
        ) from exc

    try:

        content = response["message"]["content"]

    except (
        KeyError,
        TypeError,
    ) as exc:

        raise RuntimeError(
            "Unexpected Ollama response format."
        ) from exc

    return content.strip()


# ============================================================
# DETAILED EXECUTION
# ============================================================

def run_model_detailed(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> Dict[str, Any]:
    """
    Run a model and return response plus metadata.
    """

    import time

    if not model:
        raise ValueError("model cannot be empty.")

    if not prompt:
        raise ValueError("prompt cannot be empty.")

    resolved_model = ensure_model_available(model)

    print(
        f"[Model Runner] Running model: "
        f"{resolved_model}"
    )

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

    start_time = time.perf_counter()

    try:

        response = ollama.chat(
            model=resolved_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": num_predict,
            },
        )

    except Exception as exc:

        raise RuntimeError(
            f"Ollama generation failed for "
            f"'{resolved_model}': {exc}"
        ) from exc

    latency = time.perf_counter() - start_time

    try:

        content = response["message"]["content"]

    except (
        KeyError,
        TypeError,
    ) as exc:

        raise RuntimeError(
            "Unexpected Ollama response format."
        ) from exc

    return {
        "model": resolved_model,
        "response": content.strip(),
        "latency_seconds": round(latency, 3),
        "prompt_eval_count": response.get(
            "prompt_eval_count"
        ),
        "eval_count": response.get(
            "eval_count"
        ),
        "total_duration_ns": response.get(
            "total_duration"
        ),
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("OLLAMA MODEL RUNNER TEST")
    print("=" * 60)

    print("\nInstalled models:")

    for model in get_local_models():
        print(f"  - {model}")

    print("\nModel resolution:")

    for requested in [
        "qwen3",
        "gemma:7b",
    ]:

        resolved = resolve_model_name(requested)

        print(
            f"  {requested} -> {resolved}"
        )

    print("\nRunning test...\n")

    try:

        result = run_model_detailed(
            model="qwen3",
            prompt="What is the capital of Australia?",
        )

        print(
            result["response"]
        )

        print(
            f"\nResolved model: "
            f"{result['model']}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']} seconds"
        )

    except Exception as exc:

        print("\nERROR:")
        print(exc)