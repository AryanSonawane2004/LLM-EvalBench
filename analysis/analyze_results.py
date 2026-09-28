import json
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = BASE_DIR / "data" / "benchmark.jsonl"
EVALUATIONS_FILE = BASE_DIR / "data" / "evaluations_qwen3_4b.jsonl"

SUMMARY_FILE = BASE_DIR / "data" / "summary.json"
EVALUATIONS_JSON_FILE = BASE_DIR / "data" / "evaluations.json"
CRITERION_CSV_FILE = BASE_DIR / "data" / "criterion_scores.csv"


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# HELPERS
# ============================================================

def safe_mean(values):

    return round(mean(values), 4) if values else 0.0


def get_score(record):

    return float(record.get("overall_score", 0.0))


# ============================================================
# MAIN
# ============================================================

def main():

    benchmark_records = load_jsonl(BENCHMARK_FILE)
    evaluation_records = load_jsonl(EVALUATIONS_FILE)

    if not evaluation_records:

        print("No evaluation records found.")
        return

    # --------------------------------------------------------
    # Create benchmark lookup
    # --------------------------------------------------------

    benchmark_by_id = {
        record["id"]: record
        for record in benchmark_records
    }

    # --------------------------------------------------------
    # JOIN benchmark + evaluation
    # --------------------------------------------------------

    combined_records = []

    for evaluation in evaluation_records:

        question_id = evaluation.get("id")

        benchmark = benchmark_by_id.get(question_id, {})

        combined = {
            "id": question_id,

            "category": benchmark.get("category"),
            "difficulty": benchmark.get("difficulty"),

            "question": benchmark.get("question"),
            "reference_answer": benchmark.get("reference_answer"),
            "criteria": benchmark.get("criteria", []),

            "model": evaluation.get("model"),
            "judge_model": evaluation.get("judge_model"),

            "scores": evaluation.get("scores", {}),

            "overall_score": evaluation.get(
                "overall_score",
                0.0
            ),

            "rationale": evaluation.get(
                "rationale",
                ""
            )
        }

        combined_records.append(combined)

    # ========================================================
    # OVERALL
    # ========================================================

    scores = [
        get_score(record)
        for record in combined_records
    ]

    overall = {

        "total_evaluations": len(combined_records),

        "mean_score": safe_mean(scores),

        "minimum_score": min(scores),

        "maximum_score": max(scores)
    }

    # ========================================================
    # CATEGORY
    # ========================================================

    category_data = defaultdict(list)

    for record in combined_records:

        category = record.get(
            "category",
            "unknown"
        )

        category_data[category].append(
            get_score(record)
        )

    by_category = {}

    for category, values in sorted(
        category_data.items()
    ):

        by_category[category] = {

            "count": len(values),

            "mean_score": safe_mean(values),

            "minimum_score": min(values),

            "maximum_score": max(values)
        }

    # ========================================================
    # DIFFICULTY
    # ========================================================

    difficulty_data = defaultdict(list)

    for record in combined_records:

        difficulty = record.get(
            "difficulty",
            "unknown"
        )

        difficulty_data[difficulty].append(
            get_score(record)
        )

    by_difficulty = {}

    for difficulty, values in sorted(
        difficulty_data.items()
    ):

        by_difficulty[difficulty] = {

            "count": len(values),

            "mean_score": safe_mean(values),

            "minimum_score": min(values),

            "maximum_score": max(values)
        }

    # ========================================================
    # CRITERION
    # ========================================================

    criterion_data = defaultdict(list)

    for record in combined_records:

        scores_dict = record.get(
            "scores",
            {}
        )

        for criterion, score in scores_dict.items():

            criterion_data[criterion].append(
                float(score)
            )

    by_criterion = {}

    for criterion, values in sorted(
        criterion_data.items()
    ):

        by_criterion[criterion] = {

            "count": len(values),

            "mean_score": safe_mean(values),

            "minimum_score": min(values),

            "maximum_score": max(values)
        }

    # ========================================================
    # LOWEST SCORING QUESTIONS
    # ========================================================

    lowest_questions = sorted(
        combined_records,
        key=lambda r: get_score(r)
    )[:5]

    lowest_questions_output = []

    for record in lowest_questions:

        lowest_questions_output.append({

            "id": record.get("id"),

            "category": record.get(
                "category",
                "unknown"
            ),

            "difficulty": record.get(
                "difficulty",
                "unknown"
            ),

            "overall_score": get_score(record)
        })

    # ========================================================
    # SUMMARY.JSON
    # ========================================================

    summary = {

        "model": combined_records[0].get(
            "model"
        ),

        "judge_model": combined_records[0].get(
            "judge_model"
        ),

        "overall": overall,

        "by_category": by_category,

        "by_difficulty": by_difficulty,

        "by_criterion": by_criterion,

        "lowest_scoring_questions":
            lowest_questions_output
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # EVALUATIONS.JSON
    # ========================================================

    evaluations_output = {

        "model": combined_records[0].get(
            "model"
        ),

        "judge_model": combined_records[0].get(
            "judge_model"
        ),

        "count": len(combined_records),

        "evaluations": combined_records
    }

    with open(
        EVALUATIONS_JSON_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluations_output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # CRITERION CSV
    # ========================================================

    with open(
        CRITERION_CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "id",
            "category",
            "difficulty",
            "criterion",
            "score"
        ])

        for record in combined_records:

            scores_dict = record.get(
                "scores",
                {}
            )

            for criterion, score in scores_dict.items():

                writer.writerow([

                    record.get("id"),

                    record.get(
                        "category",
                        "unknown"
                    ),

                    record.get(
                        "difficulty",
                        "unknown"
                    ),

                    criterion,

                    score
                ])

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)

    print(
        f"Total evaluations: "
        f"{overall['total_evaluations']}"
    )

    print(
        f"Mean score:        "
        f"{overall['mean_score']:.4f}"
    )

    print(
        f"Minimum score:     "
        f"{overall['minimum_score']:.4f}"
    )

    print(
        f"Maximum score:     "
        f"{overall['maximum_score']:.4f}"
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE BY CATEGORY")
    print("=" * 60)

    for category, data in by_category.items():

        print(
            f"{category:25s} "
            f"{data['count']:2d} "
            f"{data['mean_score']:.4f} "
            f"{data['minimum_score']:.1f} "
            f"{data['maximum_score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("PERFORMANCE BY DIFFICULTY")
    print("=" * 60)

    for difficulty, data in by_difficulty.items():

        print(
            f"{difficulty:10s} "
            f"{data['count']:2d} "
            f"{data['mean_score']:.4f} "
            f"{data['minimum_score']:.1f} "
            f"{data['maximum_score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("PERFORMANCE BY CRITERION")
    print("=" * 60)

    for criterion, data in by_criterion.items():

        print(
            f"{criterion:25s} "
            f"{data['count']:2d} "
            f"{data['mean_score']:.4f} "
            f"{data['minimum_score']:.1f} "
            f"{data['maximum_score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("LOWEST-SCORING QUESTIONS")
    print("=" * 60)

    for question in lowest_questions_output:

        print(
            f"{str(question['id']):15s} "
            f"{str(question['category']):22s} "
            f"{str(question['difficulty']):7s} "
            f"{question['overall_score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("SERIALIZATION")
    print("=" * 60)

    print(
        f"Summary JSON:     "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Evaluations JSON: "
        f"{EVALUATIONS_JSON_FILE}"
    )

    print(
        f"Criterion CSV:    "
        f"{CRITERION_CSV_FILE}"
    )

    print("\nSerialization complete.")


if __name__ == "__main__":
    main()