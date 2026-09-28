import json
from pathlib import Path

from llm_judge import judge_response


RESPONSES_PATH = Path(
    "data/responses.jsonl"
)


def load_first_response():

    with RESPONSES_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        first_line = file.readline()

    return json.loads(first_line)


if __name__ == "__main__":

    record = load_first_response()

    print("=" * 50)
    print("Testing LLM Judge")
    print("=" * 50)

    print("\nQuestion:")
    print(record["prompt"])

    print("\nModel response:")
    print(record["response"])

    print("\nCriteria:")
    print(record["evaluation_criteria"])

    result = judge_response(record)

    print("\nJudge result:")
    print(
        json.dumps(
            result,
            indent=2
        )
    )