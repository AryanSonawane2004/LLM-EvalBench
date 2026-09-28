

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set

import ollama
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP CONFIG
# ============================================================

app = FastAPI(
    title="LLM-EvalBench API",
    description="LLM evaluation and benchmarking platform using Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
BENCHMARK_FILE = DATA_DIR / "benchmark.jsonl"

OLLAMA_LIBRARY_URL = "https://ollama.com/library"

# session_id -> models pulled by this application session
TEMPORARY_MODELS: Dict[str, Set[str]] = {}


# ============================================================
# PYDANTIC REQUEST MODELS
# ============================================================

class SessionResponse(BaseModel):
    session_id: str


class CleanupRequest(BaseModel):
    session_id: str


class BenchmarkRequest(BaseModel):
    model: str
    judge_model: str
    session_id: Optional[str] = None


class EvaluateRequest(BaseModel):
    model: str
    judge_model: str
    question_id: str
    session_id: Optional[str] = None


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_data_directory():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_jsonl(path: Path) -> List[dict]:
    """
    Read a JSONL file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} of {path}"
                ) from exc

    return records


# ============================================================
# BENCHMARK VALIDATION
# ============================================================

ALLOWED_CATEGORIES = {
    "factual_qa",
    "reasoning",
    "mathematics",
    "coding",
    "instruction_following",
    "summarization",
}

ALLOWED_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

ALLOWED_CRITERIA = {
    "correctness",
    "relevance",
    "completeness",
    "reasoning",
    "instruction_following",
    "faithfulness",
    "groundedness",
    "hallucination",
    "logical_consistency",
}


def validate_benchmark_record(
    record: dict,
    index: int,
):
    """
    Validate the ACTUAL benchmark schema.

    Your benchmark uses:

        id
        category
        difficulty
        source
        tags
        prompt
        reference_answer
        evaluation_criteria
    """

    required_fields = {
        "id",
        "category",
        "difficulty",
        "source",
        "tags",
        "prompt",
        "reference_answer",
        "evaluation_criteria",
    }

    missing = (
        required_fields - record.keys()
    )

    if missing:

        raise ValueError(
            f"Benchmark record {index} "
            f"missing fields: "
            f"{sorted(missing)}"
        )

    if record["category"] not in ALLOWED_CATEGORIES:

        raise ValueError(
            f"Invalid category in record "
            f"{index}: "
            f"{record['category']}"
        )

    if record["difficulty"] not in ALLOWED_DIFFICULTIES:

        raise ValueError(
            f"Invalid difficulty in record "
            f"{index}: "
            f"{record['difficulty']}"
        )

    if not isinstance(
        record["tags"],
        list,
    ):

        raise ValueError(
            f"'tags' must be a list "
            f"in record {index}"
        )

    if not isinstance(
        record["evaluation_criteria"],
        list,
    ):

        raise ValueError(
            f"'evaluation_criteria' "
            f"must be a list in "
            f"record {index}"
        )

    invalid_criteria = (
        set(record["evaluation_criteria"])
        - ALLOWED_CRITERIA
    )

    if invalid_criteria:

        raise ValueError(
            f"Invalid evaluation criteria "
            f"in record {index}: "
            f"{sorted(invalid_criteria)}"
        )


def load_benchmark() -> List[dict]:
    """
    Load and validate benchmark.jsonl.
    """

    ensure_data_directory()

    records = read_jsonl(
        BENCHMARK_FILE
    )

    for index, record in enumerate(
        records,
        start=1,
    ):

        validate_benchmark_record(
            record,
            index,
        )

    return records


# ============================================================
# OLLAMA LOCAL MODELS
# ============================================================

def get_local_ollama_models() -> List[str]:
    """
    Return exact model names currently installed locally.
    """

    try:

        result = ollama.list()

    except Exception as exc:

        print(
            f"[Ollama] Could not list "
            f"local models: {exc}"
        )

        return []

    models = []

    for model in result.get(
        "models",
        [],
    ):

        name = (
            model.get("model")
            or model.get("name")
        )

        if name:
            models.append(name)

    return models


def normalize_model_name(
    name: str,
) -> str:
    """
    qwen3:4b -> qwen3
    gemma:2b -> gemma
    """

    return (
        name
        .split(":", 1)[0]
        .strip()
        .lower()
    )


def model_is_installed(
    model_name: str,
) -> bool:

    local_models = (
        get_local_ollama_models()
    )

    target = model_name.lower()

    target_base = normalize_model_name(
        model_name
    )

    for installed in local_models:

        installed_lower = (
            installed.lower()
        )

        installed_base = (
            normalize_model_name(
                installed
            )
        )

        if installed_lower == target:
            return True

        if installed_base == target_base:
            return True

    return False


# ============================================================
# MODEL CLASSIFICATION
# ============================================================

EMBEDDING_KEYWORDS = {
    "embedding",
    "embed",
    "bge",
    "e5",
    "minilm",
    "mxbai",
    "nomic-embed",
}

VISION_KEYWORDS = {
    "llava",
    "bakllava",
    "qwen-vl",
    "qwen2-vl",
    "qwen3-vl",
    "qwen3.5-vl",
    "vision",
    "minicpm-v",
}

OCR_KEYWORDS = {
    "ocr",
}

RERANKER_KEYWORDS = {
    "rerank",
    "reranker",
}

CODE_KEYWORDS = {
    "code",
    "coder",
    "codellama",
    "codegemma",
    "codestral",
    "codeqwen",
    "devstral",
}


def classify_model(
    model_name: str,
) -> dict:

    name = model_name.lower()

    if any(
        keyword in name
        for keyword in EMBEDDING_KEYWORDS
    ):

        return {
            "model_type": "embedding",
            "benchmark_compatible": False,
        }

    if any(
        keyword in name
        for keyword in RERANKER_KEYWORDS
    ):

        return {
            "model_type": "reranker",
            "benchmark_compatible": False,
        }

    if any(
        keyword in name
        for keyword in OCR_KEYWORDS
    ):

        return {
            "model_type": "ocr",
            "benchmark_compatible": False,
        }

    if any(
        keyword in name
        for keyword in VISION_KEYWORDS
    ):

        return {
            "model_type": "vision",
            "benchmark_compatible": True,
        }

    if any(
        keyword in name
        for keyword in CODE_KEYWORDS
    ):

        return {
            "model_type": "code_generation",
            "benchmark_compatible": True,
        }

    return {
        "model_type": "text_generation",
        "benchmark_compatible": True,
    }


# ============================================================
# OLLAMA CATALOG
# ============================================================

def get_ollama_catalog() -> List[dict]:
    """
    Fetch models from the official Ollama library page.

    If the website is unavailable, fall back to locally
    installed Ollama models.
    """

    try:

        response = requests.get(
            OLLAMA_LIBRARY_URL,
            timeout=20,
            headers={
                "User-Agent":
                    "LLM-EvalBench/1.0"
            },
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        print(
            f"[Ollama Catalog] "
            f"Failed: {exc}"
        )

        local_models = (
            get_local_ollama_models()
        )

        return [
            {
                "id": model,
                "name": model,
                "provider": "ollama",
                "source": "local",
                "installed": True,
                **classify_model(model),
            }
            for model in local_models
        ]

    html = response.text

    matches = re.findall(
        r'href=["\']/library/([^"\']+)["\']',
        html,
    )

    seen = set()

    model_names = []

    for model in matches:

        model = model.strip("/")

        if not model:
            continue

        if "/" in model:
            continue

        if model not in seen:

            seen.add(model)
            model_names.append(model)

    local_models = (
        get_local_ollama_models()
    )

    local_exact = {
        model.lower()
        for model in local_models
    }

    local_base = {
        normalize_model_name(model)
        for model in local_models
    }

    catalog = []

    for model in model_names:

        capabilities = classify_model(
            model
        )

        installed = (
            model.lower()
            in local_exact
            or
            normalize_model_name(model)
            in local_base
        )

        catalog.append(
            {
                "id": model,
                "name": model,
                "provider": "ollama",
                "source": "ollama_library",
                "installed": installed,
                **capabilities,
            }
        )

    # Add locally installed models that
    # aren't visible in the scraped catalog.

    catalog_ids = {
        item["id"].lower()
        for item in catalog
    }

    for local_model in local_models:

        if (
            local_model.lower()
            not in catalog_ids
        ):

            catalog.append(
                {
                    "id": local_model,
                    "name": local_model,
                    "provider": "ollama",
                    "source": "local",
                    "installed": True,
                    **classify_model(
                        local_model
                    ),
                }
            )

    return catalog


def get_catalog_model(
    model_name: str,
) -> Optional[dict]:

    catalog = get_ollama_catalog()

    target = model_name.lower()

    for model in catalog:

        if (
            model["id"].lower()
            == target
        ):

            return model

    target_base = (
        normalize_model_name(
            model_name
        )
    )

    for model in catalog:

        if (
            normalize_model_name(
                model["id"]
            )
            == target_base
        ):

            return model

    return None


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_benchmark_model(
    model_name: str,
):

    model = get_catalog_model(
        model_name
    )

    if model is None:

        if model_is_installed(
            model_name
        ):
            return

        raise HTTPException(
            status_code=404,
            detail=(
                f"Ollama model "
                f"'{model_name}' "
                f"was not found."
            ),
        )

    if not model.get(
        "benchmark_compatible",
        False,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' "
                f"is classified as "
                f"'{model.get('model_type')}' "
                f"and cannot be used "
                f"for benchmark generation "
                f"or judging."
            ),
        )


# ============================================================
# PULL MODEL
# ============================================================

def pull_model_for_session(
    model_name: str,
    session_id: Optional[str],
) -> bool:

    validate_benchmark_model(
        model_name
    )

    if model_is_installed(
        model_name
    ):

        print(
            f"[Ollama] {model_name} "
            f"already installed."
        )

        return False

    print(
        f"[Ollama] Pulling "
        f"{model_name}..."
    )

    try:

        ollama.pull(
            model_name
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to pull Ollama "
                f"model '{model_name}': "
                f"{exc}"
            ),
        )

    if session_id:

        TEMPORARY_MODELS.setdefault(
            session_id,
            set(),
        )

        TEMPORARY_MODELS[
            session_id
        ].add(model_name)

    print(
        f"[Ollama] Successfully pulled "
        f"{model_name}"
    )

    return True


# ============================================================
# CLEANUP
# ============================================================

def cleanup_session_models(
    session_id: str,
) -> dict:

    models = (
        TEMPORARY_MODELS.pop(
            session_id,
            set(),
        )
    )

    deleted = []
    failed = []

    for model_name in models:

        try:

            print(
                f"[Ollama] Deleting "
                f"temporary model: "
                f"{model_name}"
            )

            ollama.delete(
                model_name
            )

            deleted.append(
                model_name
            )

        except Exception as exc:

            print(
                f"[Ollama] Failed to "
                f"delete {model_name}: "
                f"{exc}"
            )

            failed.append(
                {
                    "model": model_name,
                    "error": str(exc),
                }
            )

    return {
        "session_id": session_id,
        "deleted": deleted,
        "failed": failed,
    }


# ============================================================
# RESULT CACHE
# ============================================================

def result_cache_path(
    model: str,
    judge_model: str,
) -> Path:

    safe_model = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        model,
    )

    safe_judge = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        judge_model,
    )

    return DATA_DIR / (
        f"benchmark_"
        f"{safe_model}"
        f"_judge_"
        f"{safe_judge}"
        f".json"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "LLM-EvalBench API",
        "version": "1.0.0",
        "status": "running",
        "provider": "ollama",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    local_models = (
        get_local_ollama_models()
    )

    return {
        "status": "healthy",
        "provider": "ollama",
        "ollama_available": True,
        "installed_models": local_models,
    }


# ============================================================
# MODELS
# ============================================================

@app.get("/api/models")
def list_models():

    catalog = get_ollama_catalog()

    compatible_count = sum(
        1
        for model in catalog
        if model.get(
            "benchmark_compatible"
        )
    )

    return {
        "provider": "ollama",
        "source": "ollama_library",
        "count": len(catalog),
        "benchmark_compatible_count":
            compatible_count,
        "models": catalog,
    }


# ============================================================
# SESSION CREATE
# ============================================================

@app.post(
    "/api/session/create",
    response_model=SessionResponse,
)
def create_session():

    session_id = str(
        uuid.uuid4()
    )

    TEMPORARY_MODELS[
        session_id
    ] = set()

    print(
        f"[Session] Created: "
        f"{session_id}"
    )

    return {
        "session_id": session_id
    }


# ============================================================
# SESSION CLEANUP
# ============================================================

@app.post(
    "/api/session/cleanup"
)
def cleanup_session(
    request: CleanupRequest,
):

    result = cleanup_session_models(
        request.session_id
    )

    print(
        f"[Session] Cleanup completed: "
        f"{request.session_id}"
    )

    return result


# ============================================================
# BENCHMARK DATA
# ============================================================

@app.get("/api/benchmark")
def get_benchmark():

    try:

        records = load_benchmark()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    return {
        "count": len(records),
        "benchmark": records,
    }


# ============================================================
# IMPORT MODEL RUNNER
# ============================================================

def import_model_runner():

    try:

        from models.model_runner import (
            run_model,
        )

        return run_model

    except ImportError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not import "
                f"model runner: {exc}"
            ),
        )


# ============================================================
# IMPORT JUDGE
# ============================================================

def import_llm_judge():

    try:

        from evaluation.llm_judge import (
            judge_response,
        )

        return judge_response

    except ImportError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not import "
                f"LLM judge: {exc}"
            ),
        )


# ============================================================
# SINGLE QUESTION EVALUATION
# ============================================================

@app.post("/api/evaluate")
def evaluate_question(
    request: EvaluateRequest,
):

    benchmark = load_benchmark()

    question = None

    for item in benchmark:

        if (
            item["id"]
            == request.question_id
        ):

            question = item
            break

    if question is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Question "
                f"'{request.question_id}' "
                f"was not found."
            ),
        )

    # Pull models if required

    pull_model_for_session(
        request.model,
        request.session_id,
    )

    pull_model_for_session(
        request.judge_model,
        request.session_id,
    )

    run_model = (
        import_model_runner()
    )

    judge_response_fn = (
        import_llm_judge()
    )

    try:

        model_response = run_model(
            model=request.model,
            prompt=question["prompt"],
        )

        evaluation = (
            judge_response_fn(
                question=question,
                model_response=model_response,
                judge_model=request.judge_model,
            )
        )

        return {
            "question": question,
            "model": request.model,
            "judge_model":
                request.judge_model,
            "model_response":
                model_response,
            "evaluation":
                evaluation,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# FULL BENCHMARK
# ============================================================

@app.post("/api/run-benchmark")
def run_benchmark(
    request: BenchmarkRequest,
):

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    benchmark = load_benchmark()

    if not benchmark:

        raise HTTPException(
            status_code=400,
            detail=(
                "Benchmark dataset "
                "is empty."
            ),
        )

    # --------------------------------------------------------
    # Validate models
    # --------------------------------------------------------

    validate_benchmark_model(
        request.model
    )

    validate_benchmark_model(
        request.judge_model
    )

    # --------------------------------------------------------
    # Pull models
    # --------------------------------------------------------

    pull_model_for_session(
        request.model,
        request.session_id,
    )

    pull_model_for_session(
        request.judge_model,
        request.session_id,
    )

    # --------------------------------------------------------
    # Load runner/judge
    # --------------------------------------------------------

    run_model = (
        import_model_runner()
    )

    judge_response_fn = (
        import_llm_judge()
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    total_score = 0.0

    scored_questions = 0

    # --------------------------------------------------------
    # Run every question
    # --------------------------------------------------------

    for index, question in enumerate(
        benchmark,
        start=1,
    ):

        question_id = question["id"]

        print(
            f"[Benchmark] "
            f"{index}/"
            f"{len(benchmark)} "
            f"{question_id}"
        )

        try:

            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            model_response = run_model(
                model=request.model,
                prompt=question["prompt"],
            )

            # ------------------------------------------------
            # Judge answer
            # ------------------------------------------------

            evaluation = (
                judge_response_fn(
                    question=question,
                    model_response=model_response,
                    judge_model=request.judge_model,
                )
            )

            # ------------------------------------------------
            # Extract calculated score
            # ------------------------------------------------

            score = None

            if isinstance(
                evaluation,
                dict,
            ):

                possible_score = (
                    evaluation.get(
                        "overall_score"
                    )
                )

                if isinstance(
                    possible_score,
                    (int, float),
                ):

                    score = float(
                        possible_score
                    )

                elif isinstance(
                    possible_score,
                    str,
                ):

                    try:

                        score = float(
                            possible_score
                        )

                    except ValueError:

                        score = None

            if score is not None:

                total_score += score

                scored_questions += 1

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            results.append(
                {
                    "question_id":
                        question_id,

                    "question":
                        question["prompt"],

                    "category":
                        question["category"],

                    "difficulty":
                        question["difficulty"],

                    "source":
                        question["source"],

                    "tags":
                        question["tags"],

                    "reference_answer":
                        question[
                            "reference_answer"
                        ],

                    "criteria":
                        question[
                            "evaluation_criteria"
                        ],

                    "model_response":
                        model_response,

                    "evaluation":
                        evaluation,

                    "error":
                        None,
                }
            )

        except Exception as exc:

            print(
                f"[Benchmark] ERROR "
                f"{question_id}: "
                f"{exc}"
            )

            results.append(
                {
                    "question_id":
                        question_id,

                    "question":
                        question["prompt"],

                    "category":
                        question["category"],

                    "difficulty":
                        question["difficulty"],

                    "source":
                        question["source"],

                    "tags":
                        question["tags"],

                    "reference_answer":
                        question[
                            "reference_answer"
                        ],

                    "criteria":
                        question[
                            "evaluation_criteria"
                        ],

                    "model_response":
                        None,

                    "evaluation":
                        None,

                    "error":
                        str(exc),
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = sum(
        1
        for result in results
        if result["error"] is None
    )

    failed = (
        len(results)
        - successful
    )

    overall_score = None

    if scored_questions > 0:

        overall_score = round(
            total_score
            / scored_questions,
            2,
        )

    summary = {
        "model":
            request.model,

        "judge_model":
            request.judge_model,

        "total_questions":
            len(benchmark),

        "successful_questions":
            successful,

        "failed_questions":
            failed,

        "scored_questions":
            scored_questions,

        "overall_score":
            overall_score,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    cache_path = result_cache_path(
        request.model,
        request.judge_model,
    )

    try:

        with cache_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[Benchmark] Saved results "
            f"to {cache_path}"
        )

    except Exception as exc:

        print(
            f"[Benchmark] Could not "
            f"save cache: {exc}"
        )

    return output


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    ensure_data_directory()

    print("=" * 60)
    print("LLM-EvalBench API")
    print("=" * 60)

    print(
        "Provider: Ollama"
    )

    print(
        f"Benchmark: "
        f"{BENCHMARK_FILE}"
    )

    try:

        local_models = (
            get_local_ollama_models()
        )

        print(
            f"Installed Ollama models: "
            f"{len(local_models)}"
        )

        for model in local_models:

            print(
                f"  - {model}"
            )

    except Exception as exc:

        print(
            f"Could not inspect "
            f"Ollama: {exc}"
        )

    print("=" * 60)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )