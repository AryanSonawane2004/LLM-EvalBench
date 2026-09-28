import json
from pathlib import Path
from typing import Dict, Any, List

from llm_judge import judge_response


RESPONSES_PATH = Path(
    "data/responses.jsonl"
)

OUTPUT_PATH = Path(
    "data/evaluation_results.jsonl"
)


def load_responses(
    path: Path
) -> List[Dict[str, Any]]:

    results = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                results.append(
                    json.loads(line)
                )

    return results


def evaluate_response(
    record: Dict[str, Any]
) -> Dict[str, Any]:

    judge_result = judge_response(record)

    return {
        "id": record["id"],
        "model": record["model"],
        "category": record["category"],
        "difficulty": record["difficulty"],
        "prompt": record["prompt"],
        "reference_answer": record[
            "reference_answer"
        ],
        "response": record["response"],
        "criteria": record[
            "evaluation_criteria"
        ],
        "scores": judge_result.get(
            "scores",
            {}
        ),
        "overall_score": judge_result.get(
            "overall_score"
        ),
        "rationale": judge_result.get(
            "rationale"
        )
    }


def main():

    records = load_responses(
        RESPONSES_PATH
    )

    evaluation_results = []

    for index, record in enumerate(
        records,
        start=1
    ):

        print(
            f"\n[{index}/{len(records)}] "
            f"Evaluating {record['id']}"
        )

        try:

            result = evaluate_response(
                record
            )

            evaluation_results.append(
                result
            )

            print(
                f"  ✓ Score: "
                f"{result['overall_score']}"
            )

        except Exception as e:

            print(
                f"  ✗ Error: {e}"
            )

            evaluation_results.append({
                "id": record["id"],
                "model": record["model"],
                "error": str(e)
            })

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        for result in evaluation_results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    successful = sum(
        1
        for result in evaluation_results
        if "error" not in result
    )

    failed = len(evaluation_results) - successful

    print("\n" + "=" * 50)
    print("LLM Evaluation Complete")
    print("=" * 50)

    print(
        f"Total responses: "
        f"{len(evaluation_results)}"
    )

    print(
        f"Successful evaluations: "
        f"{successful}"
    )

    print(
        f"Failed evaluations: "
        f"{failed}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()