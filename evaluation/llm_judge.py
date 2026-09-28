# evaluation/llm_judge.py

import json
import re
from typing import Any, Dict, List

import ollama


# ============================================================
# CONFIG
# ============================================================

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


# ============================================================
# ALLOWED CRITERIA
# ============================================================

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
# OLLAMA MODEL HELPERS
# ============================================================

def get_local_models() -> List[str]:
    """Return models currently installed in Ollama."""

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


def is_model_installed(model_name: str) -> bool:
    """Check whether an Ollama model is installed."""

    target = model_name.lower()
    target_base = target.split(":", 1)[0]

    for installed in get_local_models():

        installed_lower = installed.lower()
        installed_base = installed_lower.split(":", 1)[0]

        if installed_lower == target:
            return True

        if installed_base == target_base:
            return True

    return False


def ensure_judge_model(model_name: str):
    """
    Pull the judge model if it is not already installed.

    Session cleanup is handled by main.py.
    """

    if not model_name:
        raise ValueError(
            "judge_model cannot be empty."
        )

    if is_model_installed(model_name):

        print(
            f"[LLM Judge] {model_name} already installed."
        )

        return

    print(
        f"[LLM Judge] {model_name} is not installed."
    )

    print(
        f"[LLM Judge] Pulling {model_name}..."
    )

    try:

        ollama.pull(model_name)

    except Exception as exc:

        raise RuntimeError(
            f"Failed to pull judge model "
            f"'{model_name}': {exc}"
        ) from exc

    print(
        f"[LLM Judge] Successfully pulled "
        f"{model_name}."
    )


# ============================================================
# JUDGE PROMPT
# ============================================================

def build_judge_prompt(
    question: Dict[str, Any],
    model_response: str,
) -> str:
    """
    Build the evaluation prompt using the actual benchmark schema.

    Benchmark fields:
        prompt
        reference_answer
        evaluation_criteria
    """

    prompt = question.get(
        "prompt",
        "",
    )

    reference_answer = question.get(
        "reference_answer",
        "",
    )

    criteria = question.get(
        "evaluation_criteria",
        [],
    )

    category = question.get(
        "category",
        "",
    )

    difficulty = question.get(
        "difficulty",
        "",
    )

    criteria_text = "\n".join(
        f"- {criterion}"
        for criterion in criteria
    )

    return f"""
You are an expert LLM evaluator.

Evaluate the candidate model response against the benchmark
prompt, reference answer, and requested evaluation criteria.

Be strict and evidence-based.

Do not reward an answer simply because it sounds confident.

Do not invent facts.

For every requested criterion, provide:
1. A score from 0 to 5.
2. A concise rationale explaining the score.

Scoring scale:

0 = completely fails
1 = major problems
2 = substantial problems
3 = acceptable / partially successful
4 = strong
5 = excellent

Criterion definitions:

correctness:
Whether the answer reaches the correct conclusion.

relevance:
Whether the answer directly addresses the prompt.

completeness:
Whether the important required information is present.

reasoning:
Whether the reasoning is logically sound.

instruction_following:
Whether explicit instructions were followed.

faithfulness:
Whether the answer accurately represents the supplied information.

groundedness:
Whether claims are supported by the supplied information.

hallucination:
Whether unsupported information was introduced.

logical_consistency:
Whether the answer is internally consistent.

Benchmark category:
{category}

Benchmark difficulty:
{difficulty}

PROMPT:
{prompt}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE MODEL RESPONSE:
{model_response}

REQUESTED EVALUATION CRITERIA:
{criteria_text}

Return ONLY valid JSON.

Required structure:

{{
  "criteria_scores": {{
    "criterion_name": {{
      "score": 0,
      "rationale": "..."
    }}
  }},
  "overall_rationale": "..."
}}

Rules:

- Include every requested criterion.
- Do not include criteria that were not requested.
- Every score must be an integer from 0 to 5.
- Keep rationales concise.
- Do not use Markdown.
- Do not wrap the JSON in code fences.
""".strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from the judge response.

    Handles occasional Markdown code fences.
    """

    if not text:
        raise ValueError(
            "Judge returned an empty response."
        )

    cleaned = text.strip()

    # Remove ```json ... ``` if present
    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    # Try direct JSON first
    try:

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Try extracting the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Judge did not return valid JSON."
        )

    candidate = cleaned[start:end + 1]

    try:

        parsed = json.loads(candidate)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Could not parse judge JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Judge JSON must be an object."
        )

    return parsed


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_score(value: Any) -> int:
    """Convert judge score to integer 0-5."""

    try:

        score = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"Invalid judge score: {value}"
        ) from exc

    score = max(
        0,
        min(
            5,
            score,
        ),
    )

    return int(round(score))


# ============================================================
# VALIDATE EVALUATION
# ============================================================

def validate_evaluation(
    evaluation: Dict[str, Any],
    requested_criteria: List[str],
) -> Dict[str, Any]:
    """
    Validate judge output and calculate overall score.

    The final overall score is calculated by Python rather than
    blindly trusting a number produced by the judge.
    """

    if "criteria_scores" not in evaluation:

        raise ValueError(
            "Judge response is missing "
            "'criteria_scores'."
        )

    criteria_scores = evaluation[
        "criteria_scores"
    ]

    if not isinstance(
        criteria_scores,
        dict,
    ):

        raise ValueError(
            "'criteria_scores' must be an object."
        )

    normalized_scores = {}

    for criterion in requested_criteria:

        if criterion not in ALLOWED_CRITERIA:

            raise ValueError(
                f"Unsupported criterion: "
                f"{criterion}"
            )

        if criterion not in criteria_scores:

            raise ValueError(
                f"Judge did not score "
                f"criterion: {criterion}"
            )

        criterion_result = criteria_scores[
            criterion
        ]

        if not isinstance(
            criterion_result,
            dict,
        ):

            raise ValueError(
                f"Invalid result for "
                f"criterion: {criterion}"
            )

        if "score" not in criterion_result:

            raise ValueError(
                f"Missing score for "
                f"criterion: {criterion}"
            )

        score = normalize_score(
            criterion_result["score"]
        )

        rationale = criterion_result.get(
            "rationale",
            "",
        )

        if rationale is None:
            rationale = ""

        normalized_scores[criterion] = {
            "score": score,
            "rationale": str(
                rationale
            ).strip(),
        }

    # Calculate overall score ourselves
    scores = [
        item["score"]
        for item in normalized_scores.values()
    ]

    overall_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    overall_rationale = evaluation.get(
        "overall_rationale",
        "",
    )

    if overall_rationale is None:
        overall_rationale = ""

    return {
        "overall_score": round(
            overall_score,
            2,
        ),
        "criteria_scores": normalized_scores,
        "overall_rationale": str(
            overall_rationale
        ).strip(),
    }


# ============================================================
# MAIN JUDGE FUNCTION
# ============================================================

def judge_response(
    question: Dict[str, Any],
    model_response: str,
    judge_model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Evaluate a candidate model response using an Ollama judge.
    """

    if not question:
        raise ValueError(
            "question cannot be empty."
        )

    if not model_response:
        raise ValueError(
            "model_response cannot be empty."
        )

    if not judge_model:
        raise ValueError(
            "judge_model cannot be empty."
        )

    criteria = question.get(
        "evaluation_criteria",
        [],
    )

    if not criteria:

        raise ValueError(
            "Question does not specify "
            "evaluation_criteria."
        )

    # Validate requested criteria
    invalid_criteria = (
        set(criteria)
        - ALLOWED_CRITERIA
    )

    if invalid_criteria:

        raise ValueError(
            "Unsupported evaluation criteria: "
            f"{sorted(invalid_criteria)}"
        )

    # Make sure judge exists
    ensure_judge_model(
        judge_model
    )

    # Build judge prompt
    prompt = build_judge_prompt(
        question=question,
        model_response=model_response,
    )

    print(
        f"[LLM Judge] "
        f"Evaluating with {judge_model}"
    )

    # Call Ollama
    try:

        response = ollama.chat(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict LLM "
                        "evaluation judge. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            format="json",
        )

    except Exception as exc:

        raise RuntimeError(
            f"Ollama judge failed for "
            f"'{judge_model}': {exc}"
        ) from exc

    # Extract response text
    try:

        raw_content = response[
            "message"
        ]["content"]

    except (
        KeyError,
        TypeError,
    ) as exc:

        raise RuntimeError(
            "Unexpected Ollama judge "
            "response format."
        ) from exc

    # Parse JSON
    parsed = extract_json(
        raw_content
    )

    # Validate
    evaluation = validate_evaluation(
        evaluation=parsed,
        requested_criteria=criteria,
    )

    # Add metadata
    evaluation["judge_model"] = (
        judge_model
    )

    return evaluation


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    TEST_QUESTION = {
        "id": "fact_001",
        "category": "factual_qa",
        "difficulty": "easy",
        "source": "custom",
        "tags": [
            "geography",
            "basic_knowledge",
        ],
        "prompt": (
            "What is the capital of Australia?"
        ),
        "reference_answer": "Canberra.",
        "evaluation_criteria": [
            "correctness",
            "relevance",
        ],
    }

    TEST_RESPONSE = (
        "The capital of Australia is Canberra."
    )

    TEST_JUDGE = "gemma:7b"

    print("=" * 60)
    print("OLLAMA LLM JUDGE TEST")
    print("=" * 60)

    try:

        result = judge_response(
            question=TEST_QUESTION,
            model_response=TEST_RESPONSE,
            judge_model=TEST_JUDGE,
        )

        print()
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        print()
        print("ERROR:")
        print(exc)