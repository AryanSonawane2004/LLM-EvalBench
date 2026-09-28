import json
from pathlib import Path

import ollama


JUDGE_MODEL = "gemma:2b"

INPUT_PATH = Path( "data/judge_reliability_qwen_results.jsonl")
OUTPUT_PATH = Path("data/judge_reliability_results.jsonl")

ALLOWED_SCORES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


JUDGE_PROMPT = """
You are a strict expert evaluator.

Your task is to determine whether a model response is correct
with respect to the question and reference answer.

Follow this evaluation process:

STEP 1 — Determine the expected answer
Understand the question and the reference answer.

STEP 2 — Inspect the model response
Identify the factual, mathematical, logical, or instruction-following
claims made by the response.

STEP 3 — Compare the response with the expected answer
Determine whether the response is correct or incorrect.

STEP 4 — Identify concrete errors
If the response is incorrect, explicitly identify the incorrect claim.
Do not give credit merely because the response is fluent or plausible.

STEP 5 — Evaluate the requested criteria
Score every criterion independently.

IMPORTANT RULES:

1. Evaluate ONLY the provided criteria.
2. Use the exact criterion names provided.
3. Every criterion must receive exactly one score.
4. Use ONLY these scores:
   1.0, 0.8, 0.6, 0.4, 0.2, 0.0
5. Never use 0.9, 0.75, 0.7, 0.5, or any other value.
6. Do not reward fluent writing when the underlying answer is wrong.
7. Check calculations explicitly when mathematics is involved.
8. Check factual claims against the reference answer.
9. Check logical reasoning rather than merely checking whether the final
   answer looks plausible.
10. For instruction-following, check the instructions literally.
11. Count sentences and bullets when requested.
12. If a specific word is forbidden, check the exact word.
13. Do not treat synonyms as violations unless explicitly prohibited.
14. If the response is clearly incorrect, the correctness score must
    reflect that error.
15. Do not assume that the model response is correct.

VERDICT:

Return exactly one of:

"correct"
"partially_correct"
"incorrect"

Use:
- correct: the response satisfies the requested task and is factually correct.
- partially_correct: some important parts are correct but there is a meaningful
  error or omission.
- incorrect: the central answer or required instruction is wrong.

Return ONLY valid JSON.

Required format:

{
    "verdict": "correct",
    "scores": {
        "<exact criterion name>": 1.0
    },
    "errors": [],
    "rationale": "Brief explanation."
}
"""


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_response(
    question,
    reference,
    response,
    criteria
):
    score_properties = {
        criterion: {
            "type": "number",
            "enum": ALLOWED_SCORES
        }
        for criterion in criteria
    }

    schema = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "correct",
                    "partially_correct",
                    "incorrect"
                ]
            },
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": criteria
            },
            "errors": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "rationale": {
                "type": "string"
            }
        },
        "required": [
            "verdict",
            "scores",
            "errors",
            "rationale"
        ]
    }

    user_prompt = f"""
QUESTION:
{question}

REFERENCE ANSWER:
{reference}

MODEL RESPONSE:
{response}

EVALUATION CRITERIA:
{json.dumps(criteria, indent=2)}

Evaluate the model response using the required evaluation process.
"""

    result = ollama.chat(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": JUDGE_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format=schema
    )

    evaluation = json.loads(
        result["message"]["content"]
    )

    # ---------------------------------------------------------
    # Validate criteria
    # ---------------------------------------------------------

    returned_criteria = set(
        evaluation["scores"].keys()
    )

    expected_criteria = set(criteria)

    if returned_criteria != expected_criteria:
        raise ValueError(
            f"Incorrect criteria returned. "
            f"Expected: {sorted(expected_criteria)}, "
            f"Got: {sorted(returned_criteria)}"
        )

    # ---------------------------------------------------------
    # Validate score values
    # ---------------------------------------------------------

    for criterion, score in evaluation["scores"].items():

        if score not in ALLOWED_SCORES:
            raise ValueError(
                f"Invalid score for {criterion}: {score}"
            )

    # ---------------------------------------------------------
    # Python calculates overall score
    # ---------------------------------------------------------

    overall_score = (
        sum(evaluation["scores"].values())
        / len(criteria)
    )

    evaluation["overall_score"] = round(
        overall_score,
        4
    )

    return evaluation


def main():

    tests = load_jsonl(INPUT_PATH)

    print("=" * 60)
    print("IMPROVED LLM JUDGE RELIABILITY TEST")
    print("=" * 60)

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Test cases : {len(tests)}")
    print()

    results = []

    for index, test in enumerate(tests, start=1):

        print(
            f"[{index}/{len(tests)}] "
            f"{test['id']}"
        )

        # -----------------------------------------------------
        # Evaluate GOOD response
        # -----------------------------------------------------

        good = evaluate_response(
            question=test["question"],
            reference=test["reference_answer"],
            response=test["good_response"],
            criteria=test["criteria"]
        )

        # -----------------------------------------------------
        # Evaluate BAD response
        # -----------------------------------------------------

        bad = evaluate_response(
            question=test["question"],
            reference=test["reference_answer"],
            response=test["bad_response"],
            criteria=test["criteria"]
        )

        # -----------------------------------------------------
        # Expected labels
        # -----------------------------------------------------

        expected_good_verdict = "correct"

        expected_bad_verdict = "incorrect"

        good_verdict_correct = (
            good["verdict"] == expected_good_verdict
        )

        bad_verdict_correct = (
            bad["verdict"] == expected_bad_verdict
        )

        # -----------------------------------------------------
        # Score gap
        # -----------------------------------------------------

        score_gap = round(
            good["overall_score"]
            - bad["overall_score"],
            4
        )

        result = {
            "id": test["id"],
            "criteria": test["criteria"],

            "good_verdict": good["verdict"],
            "bad_verdict": bad["verdict"],

            "good_verdict_correct":
                good_verdict_correct,

            "bad_verdict_correct":
                bad_verdict_correct,

            "good_score":
                good["overall_score"],

            "bad_score":
                bad["overall_score"],

            "score_gap":
                score_gap,

            "good_scores":
                good["scores"],

            "bad_scores":
                bad["scores"],

            "good_errors":
                good["errors"],

            "bad_errors":
                bad["errors"],

            "good_rationale":
                good["rationale"],

            "bad_rationale":
                bad["rationale"]
        }

        results.append(result)

        print(
            f"  Good verdict: "
            f"{good['verdict']}"
        )

        print(
            f"  Bad verdict : "
            f"{bad['verdict']}"
        )

        print(
            f"  Good score  : "
            f"{good['overall_score']}"
        )

        print(
            f"  Bad score   : "
            f"{bad['overall_score']}"
        )

        print(
            f"  Score gap   : "
            f"{score_gap}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Calculate reliability metrics
    # ---------------------------------------------------------

    good_scores = [
        r["good_score"]
        for r in results
    ]

    bad_scores = [
        r["bad_score"]
        for r in results
    ]

    gaps = [
        r["score_gap"]
        for r in results
    ]

    good_verdict_accuracy = (
        sum(
            r["good_verdict_correct"]
            for r in results
        )
        / len(results)
    )

    bad_verdict_accuracy = (
        sum(
            r["bad_verdict_correct"]
            for r in results
        )
        / len(results)
    )

    combined_verdict_accuracy = (
        sum(
            r["good_verdict_correct"]
            for r in results
        )
        +
        sum(
            r["bad_verdict_correct"]
            for r in results
        )
    ) / (len(results) * 2)

    positive_gap_rate = (
        sum(
            gap > 0
            for gap in gaps
        )
        / len(gaps)
    )

    average_good = (
        sum(good_scores)
        / len(good_scores)
    )

    average_bad = (
        sum(bad_scores)
        / len(bad_scores)
    )

    average_gap = (
        sum(gaps)
        / len(gaps)
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("RELIABILITY SUMMARY")
    print("=" * 60)

    print(
        f"Average good-response score : "
        f"{average_good:.4f}"
    )

    print(
        f"Average bad-response score  : "
        f"{average_bad:.4f}"
    )

    print(
        f"Average score gap           : "
        f"{average_gap:.4f}"
    )

    print(
        f"Positive-gap rate           : "
        f"{positive_gap_rate:.4f}"
    )

    print(
        f"Good verdict accuracy       : "
        f"{good_verdict_accuracy:.4f}"
    )

    print(
        f"Bad verdict accuracy        : "
        f"{bad_verdict_accuracy:.4f}"
    )

    print(
        f"Overall verdict accuracy    : "
        f"{combined_verdict_accuracy:.4f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()