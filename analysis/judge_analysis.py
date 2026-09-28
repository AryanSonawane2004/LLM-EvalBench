import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/judge_reliability_results.jsonl"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    results = load_jsonl(INPUT_PATH)

    rows = []

    for r in results:

        # Good response
        rows.append({
            "id": r["id"],
            "actual": "correct",
            "predicted": r["good_verdict"],
            "score": r["good_score"]
        })

        # Bad response
        rows.append({
            "id": r["id"],
            "actual": "incorrect",
            "predicted": r["bad_verdict"],
            "score": r["bad_score"]
        })

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    matrix = pd.crosstab(
        df["actual"],
        df["predicted"]
    )

    print()
    print("=" * 60)
    print("JUDGE CONFUSION MATRIX")
    print("=" * 60)

    print(matrix)

    # ---------------------------------------------------------
    # Binary metrics
    # ---------------------------------------------------------

    tp = len(
        df[
            (df["actual"] == "correct") &
            (df["predicted"] == "correct")
        ]
    )

    fn = len(
        df[
            (df["actual"] == "correct") &
            (df["predicted"] == "incorrect")
        ]
    )

    fp = len(
        df[
            (df["actual"] == "incorrect") &
            (df["predicted"] == "correct")
        ]
    )

    tn = len(
        df[
            (df["actual"] == "incorrect") &
            (df["predicted"] == "incorrect")
        ]
    )

    accuracy = (
        (tp + tn) / len(df)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print()
    print("=" * 60)
    print("BINARY JUDGE METRICS")
    print("=" * 60)

    print(f"True positives : {tp}")
    print(f"False negatives: {fn}")
    print(f"False positives: {fp}")
    print(f"True negatives : {tn}")

    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    # ---------------------------------------------------------
    # Error cases
    # ---------------------------------------------------------

    errors = df[
        df["actual"] != df["predicted"]
    ]

    print()
    print("=" * 60)
    print("MISCLASSIFIED CASES")
    print("=" * 60)

    if len(errors) == 0:
        print("No misclassifications.")

    else:
        print(
            errors.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()