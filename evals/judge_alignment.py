"""Compare the LLM judge's labels with human labels and report alignment."""

import csv
import json
from pathlib import Path


EVALS_DIR = Path(__file__).resolve().parent
JUDGED_FILE = EVALS_DIR / "results_judged.json"
HUMAN_CSV_FILE = EVALS_DIR / "results_labeled.csv"
LEGACY_HUMAN_FILE = EVALS_DIR / "results_labeled.json"


def load_results(file_path):
    with file_path.open(encoding="utf-8") as file:
        if file_path.suffix.lower() == ".csv":
            return list(csv.DictReader(file))
        return json.load(file)


def index_by_question(results, file_path):
    by_question = {row["question"]: row for row in results}
    if len(by_question) != len(results):
        raise ValueError(f"Duplicate questions found in {file_path.name}")
    return by_question


def main():
    judged_results = load_results(JUDGED_FILE)
    human_file = (
        HUMAN_CSV_FILE if HUMAN_CSV_FILE.exists() else LEGACY_HUMAN_FILE
    )
    human_results = load_results(human_file)

    judge_by_question = index_by_question(judged_results, JUDGED_FILE)
    human_by_question = index_by_question(human_results, human_file)

    if judge_by_question.keys() != human_by_question.keys():
        raise ValueError(
            "The judged and human-labeled files do not contain the same questions"
        )

    tp = fp = fn = tn = 0
    skipped = 0

    for question, judge_row in judge_by_question.items():
        judge_label = judge_row.get("judge_label")
        human_label = human_by_question[question].get("label")

        if human_label is None or (
            isinstance(human_label, str) and not human_label.strip()
        ):
            skipped += 1
            continue

        if isinstance(human_label, str):
            human_label = human_label.strip().lower()

        if judge_label not in {"good", "bad"}:
            raise ValueError(f"Missing or invalid judge label for: {question}")
        if human_label not in {"good", "bad"}:
            raise ValueError(f"Missing or invalid human label for: {question}")

        print(f"\nQuestion: {question}")
        print(f"Human label: {human_label}")
        print(f"Judge label: {judge_label}")
        print(
            "Judge reasoning: "
            f"{judge_row.get('judge_reasoning', 'No reasoning provided')}"
        )

        # Treat "bad" as the positive class, matching the requested metrics.
        if judge_label == "bad" and human_label == "bad":
            tp += 1
        elif judge_label == "bad" and human_label == "good":
            fp += 1
        elif judge_label == "good" and human_label == "bad":
            fn += 1
        else:  # judge_label == "good" and human_label == "good"
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(f"\nResults compared: {total}")
    print(f"Unlabeled human results skipped: {skipped}")
    print(f"TP: {tp}  FP: {fp}")
    print(f"FN: {fn}  TN: {tn}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")


if __name__ == "__main__":
    main()
