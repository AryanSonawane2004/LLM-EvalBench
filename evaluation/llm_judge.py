# evaluation/llm_judge.py

import json
import re
from typing import Any, Dict, List

from evaluation.deterministic_checks import (
    check_instruction_constraints,
)
from models.model_runner import run_model


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
# JUDGE PROMPT
# ============================================================

def build_judge_prompt(
    question: Dict[str, Any],
    model_response: str,
    deterministic_checks: Dict[str, Any] | None = None,
) -> str:

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

    deterministic_evidence = (
        "No deterministic checks were required."
    )

    if deterministic_checks:

        deterministic_evidence = json.dumps(
            deterministic_checks,
            indent=2,
            ensure_ascii=False,
        )

    return f"""
You are an expert LLM evaluator.

Evaluate the candidate model response against the benchmark
prompt, reference answer, requested evaluation criteria,
and deterministic validation evidence.

Be strict and evidence-based.

Do not reward an answer simply because it sounds confident.

Do not invent facts.

IMPORTANT:
When deterministic validation reports that an explicit
instruction was violated, treat that result as authoritative
evidence for the instruction_following criterion.

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
Use deterministic validation evidence when available.

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

DETERMINISTIC VALIDATION:
{deterministic_evidence}

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

def extract_json(
    text: str,
) -> Dict[str, Any]:

    if not text:

        raise ValueError(
            "Judge returned an empty response."
        )

    cleaned = text.strip()

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

    try:

        parsed = json.loads(
            cleaned
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "Judge did not return valid JSON."
        )

    candidate = cleaned[
        start:end + 1
    ]

    try:

        parsed = json.loads(
            candidate
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Could not parse judge JSON: {exc}"
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):

        raise ValueError(
            "Judge JSON must be an object."
        )

    return parsed


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_score(
    value: Any,
) -> int:

    try:

        score = float(
            value
        )

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

    return int(
        round(score)
    )


# ============================================================
# VALIDATE EVALUATION
# ============================================================

def validate_evaluation(
    evaluation: Dict[str, Any],
    requested_criteria: List[str],
) -> Dict[str, Any]:

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

        criterion_result = (
            criteria_scores[
                criterion
            ]
        )

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

        normalized_scores[
            criterion
        ] = {
            "score": score,
            "rationale": str(
                rationale
            ).strip(),
        }

    scores = [
        item["score"]
        for item in (
            normalized_scores.values()
        )
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
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:

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

    if not api_key:

        raise ValueError(
            "api_key cannot be empty."
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

    invalid_criteria = (
        set(criteria)
        - ALLOWED_CRITERIA
    )

    if invalid_criteria:

        raise ValueError(
            "Unsupported evaluation criteria: "
            f"{sorted(invalid_criteria)}"
        )

    # --------------------------------------------------------
    # Deterministic validation
    # --------------------------------------------------------

    deterministic_checks = (
        check_instruction_constraints(
            question=question,
            response=model_response,
        )
    )

    # --------------------------------------------------------
    # Build judge prompt
    # --------------------------------------------------------

    prompt = build_judge_prompt(
        question=question,
        model_response=model_response,
        deterministic_checks=(
            deterministic_checks
        ),
    )

    print(
        f"[LLM Judge] "
        f"OpenRouter model={judge_model}"
    )

    # --------------------------------------------------------
    # Call OpenRouter
    # --------------------------------------------------------

    try:

        raw_content = run_model(
            model=judge_model,
            prompt=prompt,
            api_key=api_key,
            system_prompt=(
                "You are a strict LLM "
                "evaluation judge. "
                "Return ONLY a valid JSON object. "
                "Never return Markdown or explanations "
                "outside the JSON object."
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_object"
            },
        )

    except Exception as exc:

        raise RuntimeError(
            f"OpenRouter judge failed for "
            f"'{judge_model}': {exc}"
        ) from exc

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    parsed = extract_json(
        raw_content
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    evaluation = validate_evaluation(
        evaluation=parsed,
        requested_criteria=criteria,
    )

    # --------------------------------------------------------
    # Deterministic enforcement
    # --------------------------------------------------------

    if (
        "instruction_following" in criteria
        and not deterministic_checks[
            "passed"
        ]
    ):

        failed_checks = []

        for (
            check_name,
            check_result,
        ) in (
            deterministic_checks[
                "checks"
            ].items()
        ):

            if not check_result.get(
                "passed",
                True,
            ):

                failed_checks.append(
                    check_name
                )

        if failed_checks:

            rationale = (
                "The response failed explicit "
                "machine-checkable instruction "
                "constraints: "
                + ", ".join(
                    failed_checks
                )
                + "."
            )

        else:

            rationale = (
                "The response failed one or "
                "more explicit machine-checkable "
                "instruction constraints."
            )

        evaluation[
            "criteria_scores"
        ][
            "instruction_following"
        ] = {
            "score": 0,
            "rationale": rationale,
        }

        scores = [
            item["score"]
            for item in (
                evaluation[
                    "criteria_scores"
                ].values()
            )
        ]

        evaluation[
            "overall_score"
        ] = round(
            sum(scores) / len(scores),
            2,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    evaluation[
        "judge_model"
    ] = judge_model

    evaluation[
        "provider"
    ] = "openrouter"

    evaluation[
        "deterministic_checks"
    ] = deterministic_checks

    return evaluation