# main.py

import json
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    ALLOWED_ORIGINS,
    APP_ENV,
    APP_NAME,
    APP_VERSION,
    ARCHITECTURE,
    BENCHMARK_FILE,
    PROVIDER,
)


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_PROVIDER = "openrouter"

ALLOWED_CATEGORIES = {
    "factual_qa",
    "reasoning",
    "math",
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


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print(f"{APP_NAME} v{APP_VERSION}")
    print(f"Environment: {APP_ENV}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Provider: {PROVIDER}")
    print("=" * 60)

    yield

    print("LLM-EvalBench shutting down.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "LLM evaluation and benchmarking platform "
        "powered by OpenRouter."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
    ],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class BenchmarkRequest(BaseModel):
    model: str
    judge_model: str


class EvaluateRequest(BaseModel):
    question_id: str
    model: str
    judge_model: str


# ============================================================
# MODEL RUNNER IMPORTS
# ============================================================

def get_model_runner():

    from models.model_runner import (
        list_models,
        run_model,
        validate_model,
        verify_api_key,
    )

    return (
        list_models,
        run_model,
        validate_model,
        verify_api_key,
    )


# ============================================================
# JUDGE IMPORT
# ============================================================

def get_judge():

    from evaluation.llm_judge import (
        judge_response,
    )

    return judge_response


# ============================================================
# API KEY VALIDATION
# ============================================================

def get_api_key(
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> str:
    """
    Extract and validate the user's OpenRouter API key.

    The key is supplied at runtime and is never persisted.
    """

    if not x_api_key:

        raise HTTPException(
            status_code=401,
            detail=(
                "OpenRouter API key is required."
            ),
        )

    api_key = x_api_key.strip()

    if not api_key:

        raise HTTPException(
            status_code=401,
            detail=(
                "OpenRouter API key cannot be empty."
            ),
        )

    return api_key


# ============================================================
# BENCHMARK LOADING
# ============================================================

def load_benchmark() -> List[dict]:
    """
    Load the benchmark JSONL file.
    """

    if not BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            f"Benchmark file not found: "
            f"{BENCHMARK_FILE}"
        )

    questions = []

    with open(
        BENCHMARK_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                question = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on benchmark "
                    f"line {line_number}: {exc}"
                ) from exc

            questions.append(question)

    return questions


# ============================================================
# BENCHMARK VALIDATION
# ============================================================

def validate_benchmark_question(
    question: dict,
) -> None:
    """
    Validate the structure of a benchmark question.
    """

    required_fields = {
        "id",
        "category",
        "difficulty",
        "prompt",
        "reference_answer",
        "evaluation_criteria",
    }

    missing_fields = (
        required_fields
        - set(question.keys())
    )

    if missing_fields:

        raise ValueError(
            f"Question '{question.get('id')}' "
            f"is missing fields: "
            f"{sorted(missing_fields)}"
        )

    category = question["category"]

    if category not in ALLOWED_CATEGORIES:

        raise ValueError(
            f"Question '{question['id']}' "
            f"has unsupported category: "
            f"{category}"
        )

    difficulty = question["difficulty"]

    if difficulty not in ALLOWED_DIFFICULTIES:

        raise ValueError(
            f"Question '{question['id']}' "
            f"has unsupported difficulty: "
            f"{difficulty}"
        )

    criteria = question[
        "evaluation_criteria"
    ]

    if not isinstance(
        criteria,
        list,
    ):

        raise ValueError(
            f"Question '{question['id']}' "
            f"evaluation_criteria must be a list."
        )

    invalid_criteria = (
        set(criteria)
        - ALLOWED_CRITERIA
    )

    if invalid_criteria:

        raise ValueError(
            f"Question '{question['id']}' "
            f"has unsupported criteria: "
            f"{sorted(invalid_criteria)}"
        )


def validate_entire_benchmark(
    benchmark: List[dict],
) -> None:
    """
    Validate every benchmark question.
    """

    for question in benchmark:

        validate_benchmark_question(
            question
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    """
    Basic application information.
    """

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "environment": APP_ENV,
        "architecture": ARCHITECTURE,
        "provider": PROVIDER,
        "status": "running",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Health endpoint used by Render and the frontend.
    """

    return {
        "status": "ok",
        "environment": APP_ENV,
        "architecture": ARCHITECTURE,
        "provider": PROVIDER,
    }


# ============================================================
# VERIFY OPENROUTER API KEY
# ============================================================

@app.post("/api/verify")
def verify_openrouter_api_key(
    x_api_key: Optional[str] = Header(
        default=None,
    ),
):
    """
    Verify the user's OpenRouter API key.

    We verify the key by requesting the OpenRouter
    model catalog.

    The API key is never stored.
    """

    api_key = get_api_key(
        x_api_key
    )

    (
        _list_models,
        _run_model,
        _validate_model,
        verify_api_key,
    ) = get_model_runner()

    try:

        result = verify_api_key(
            api_key=api_key
        )

        return result

    except PermissionError as exc:

        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


# ============================================================
# GET OPENROUTER MODEL CATALOG
# ============================================================

@app.get("/api/models")
def get_models(
    x_api_key: Optional[str] = Header(
        default=None,
    ),
):
    """
    Retrieve the current OpenRouter model catalog.

    The frontend only calls this after API-key verification.
    """

    api_key = get_api_key(
        x_api_key
    )

    (
        list_models,
        _run_model,
        _validate_model,
        _verify_api_key,
    ) = get_model_runner()

    try:

        models = list_models(
            api_key=api_key
        )

        return {
            "provider": PROVIDER,
            "total_models": len(
                models
            ),
            "models": models,
        }

    except PermissionError as exc:

        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


# ============================================================
# GET BENCHMARK
# ============================================================

@app.get("/api/benchmark")
def get_benchmark():
    """
    Return the benchmark questions.

    This endpoint does not require an API key because
    the benchmark itself is public application data.
    """

    try:

        benchmark = load_benchmark()

        validate_entire_benchmark(
            benchmark
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "total_questions": len(
            benchmark
        ),
        "questions": benchmark,
    }


# ============================================================
# EVALUATE SINGLE QUESTION
# ============================================================

@app.post("/api/evaluate")
def evaluate_question(
    request: EvaluateRequest,
    x_api_key: Optional[str] = Header(
        default=None,
    ),
):
    """
    Run one benchmark question:

        Candidate model
              ↓
        Candidate response
              ↓
          LLM Judge
              ↓
        Deterministic checks
              ↓
           Evaluation
    """

    api_key = get_api_key(
        x_api_key
    )

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    try:

        benchmark = load_benchmark()

        validate_entire_benchmark(
            benchmark
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Find question
    # --------------------------------------------------------

    question = next(
        (
            item
            for item in benchmark
            if item.get("id")
            == request.question_id
        ),
        None,
    )

    if question is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Question "
                f"'{request.question_id}' "
                f"not found."
            ),
        )

    # --------------------------------------------------------
    # Get model runner
    # --------------------------------------------------------

    (
        _list_models,
        run_model,
        validate_model,
        _verify_api_key,
    ) = get_model_runner()

    # --------------------------------------------------------
    # Validate candidate + judge
    # --------------------------------------------------------

    try:

        validate_model(
            model=request.model,
            api_key=api_key,
            require_benchmark_compatible=True,
        )

        validate_model(
            model=request.judge_model,
            api_key=api_key,
            require_benchmark_compatible=True,
        )

    except PermissionError as exc:

        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Candidate model
    # --------------------------------------------------------

    try:

        model_response = run_model(
            model=request.model,
            prompt=question["prompt"],
            api_key=api_key,
        )

    except PermissionError as exc:

        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Candidate model failed: "
                f"{exc}"
            ),
        ) from exc

    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------

    try:

        judge_response = get_judge()

        evaluation = judge_response(
            question=question,
            model_response=model_response,
            judge_model=request.judge_model,
            api_key=api_key,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Judge model failed: "
                f"{exc}"
            ),
        ) from exc

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "question_id": (
            request.question_id
        ),
        "provider": PROVIDER,
        "model": request.model,
        "judge_model": (
            request.judge_model
        ),
        "model_response": model_response,
        "evaluation": evaluation,
    }


# ============================================================
# RUN FULL BENCHMARK
# ============================================================

@app.post("/api/run-benchmark")
def run_benchmark(
    request: BenchmarkRequest,
    api_key: str = Depends(get_api_key),
):
    benchmark = load_benchmark()

    if not benchmark:
        raise HTTPException(
            status_code=400,
            detail="Benchmark dataset is empty.",
        )

    # ---------------------------------------------------------
    # Get model runner functions
    # ---------------------------------------------------------

    (
        _list_models,
        run_model,
        validate_model,
        _verify_api_key,
    ) = get_model_runner()

    # ---------------------------------------------------------
    # Get LLM judge function
    # ---------------------------------------------------------

    judge_response = get_judge()

    # ---------------------------------------------------------
    # Validate candidate + judge models before running 20 items
    # ---------------------------------------------------------

    try:
        validate_model(
            model=request.model,
            api_key=api_key,
            require_benchmark_compatible=True,
        )

        validate_model(
            model=request.judge_model,
            api_key=api_key,
            require_benchmark_compatible=True,
        )

    except PermissionError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model validation failed: {exc}",
        ) from exc

    results = []
    successful_results = []
    failed_results = []

    for question in benchmark:
        question_id = question["id"]

        try:
            # Generate candidate response
            model_response = run_model(
                request.model,
                question["prompt"],
                api_key,
            )

            # Evaluate candidate response
            evaluation = judge_response(
                question=question,
                model_response=model_response,
                judge_model=request.judge_model,
                api_key=api_key,
            )

            result = {
                "id": question_id,
                "category": question.get("category"),
                "difficulty": question.get("difficulty"),
                "model": request.model,
                "judge_model": request.judge_model,
                "provider": PROVIDER,
                "model_response": model_response,
                "evaluation": evaluation,
                "status": "success",
            }

            results.append(result)
            successful_results.append(result)

        except Exception as exc:
            error_message = str(exc)

            # Classify common infrastructure failures
            if "Insufficient credits" in error_message:
                error_type = "insufficient_credits"
            elif (
                isinstance(exc, PermissionError)
                or "invalid, expired" in error_message.lower()
                or "unauthorized" in error_message.lower()
                or "permission to access" in error_message.lower()
                or "401" in error_message
            ):
                error_type = "authentication_error"
            elif "429" in error_message or "rate limit" in error_message.lower():
                error_type = "rate_limit"
            elif "timeout" in error_message.lower():
                error_type = "timeout"
            else:
                error_type = "evaluation_error"

            result = {
                "id": question_id,
                "category": question.get("category"),
                "difficulty": question.get("difficulty"),
                "model": request.model,
                "judge_model": request.judge_model,
                "provider": PROVIDER,
                "status": "failed",
                "error_type": error_type,
                "error": error_message,
            }

            results.append(result)
            failed_results.append(result)

    total_questions = len(benchmark)
    successful_count = len(successful_results)
    failed_count = len(failed_results)

    # ---------------------------------------------------------
    # IMPORTANT:
    # Only calculate benchmark-level scores when ALL questions
    # have been evaluated successfully.
    # ---------------------------------------------------------

    benchmark_status = (
        "completed"
        if successful_count == total_questions
        else "partial"
    )

    overall_score = None
    by_category = {}
    by_difficulty = {}
    by_criterion = {}
    lowest_scoring = []

    if successful_count == total_questions:
        scores = [
            result["evaluation"]["overall_score"]
            for result in successful_results
        ]

        overall_score = round(sum(scores) / len(scores), 2)

        # -------------------------
        # Category aggregation
        # -------------------------
        category_scores = {}

        for result in successful_results:
            category = result.get("category", "unknown")
            score = result["evaluation"]["overall_score"]

            category_scores.setdefault(category, []).append(score)

        by_category = {
            category: round(sum(scores) / len(scores), 2)
            for category, scores in category_scores.items()
        }

        # -------------------------
        # Difficulty aggregation
        # -------------------------
        difficulty_scores = {}

        for result in successful_results:
            difficulty = result.get("difficulty", "unknown")
            score = result["evaluation"]["overall_score"]

            difficulty_scores.setdefault(difficulty, []).append(score)

        by_difficulty = {
            difficulty: round(sum(scores) / len(scores), 2)
            for difficulty, scores in difficulty_scores.items()
        }

        # -------------------------
        # Criterion aggregation
        # -------------------------
        criterion_scores = {}

        for result in successful_results:
            criteria_scores = result["evaluation"].get(
                "criteria_scores",
                {},
            )

            for criterion, criterion_data in criteria_scores.items():
                score = criterion_data["score"]

                criterion_scores.setdefault(
                    criterion,
                    [],
                ).append(score)

        by_criterion = {
            criterion: round(sum(scores) / len(scores), 2)
            for criterion, scores in criterion_scores.items()
        }

        # -------------------------
        # Lowest scoring questions
        # -------------------------
        lowest_scoring = sorted(
            [
                {
                    "id": result["id"],
                    "category": result.get("category"),
                    "difficulty": result.get("difficulty"),
                    "score": result["evaluation"]["overall_score"],
                }
                for result in successful_results
            ],
            key=lambda item: item["score"],
        )[:5]

    # ---------------------------------------------------------
    # Failure summary
    # ---------------------------------------------------------

    error_summary = {}

    for result in failed_results:
        error_type = result.get(
            "error_type",
            "evaluation_error",
        )

        error_summary[error_type] = (
            error_summary.get(error_type, 0) + 1
        )

    return {
        "source": "runtime",
        "provider": PROVIDER,
        "model": request.model,
        "judge_model": request.judge_model,

        "status": benchmark_status,

        "total_questions": total_questions,
        "successful": successful_count,
        "failed": failed_count,

        "overall_score": overall_score,

        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_criterion": by_criterion,

        "lowest_scoring": lowest_scoring,

        "error_summary": error_summary,

        "results": results,
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )