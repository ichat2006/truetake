"""Compare saved cards with human labels without tuning or inventing ground truth."""

import argparse
import csv
import json
from pathlib import Path


def evaluate_cards(batch_path: str, labels_path: str, split: str) -> dict:
    cards = json.loads(Path(batch_path).read_text())["cards"]
    with Path(labels_path).open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"clip_id", "split", "goal_met"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Labels require clip_id, split, and goal_met columns.")
        rows = [row for row in reader if row["split"] == split]
    seen = set()
    result = {"split": split, "labeled_clips": len(rows), "missing_cards": [],
              "goal_correct": 0, "goal_wrong": 0, "goal_inconclusive": 0, "goal_unlabeled": 0}
    for row in rows:
        clip = row["clip_id"]
        if clip in seen:
            raise ValueError(f"Duplicate label for {clip}")
        seen.add(clip)
        if clip not in cards:
            result["missing_cards"].append(clip)
            continue
        truth = {"yes": True, "no": False}.get(row["goal_met"])
        if truth is None:
            result["goal_unlabeled"] += 1
            continue
        check = next((check for check in cards[clip]["checks"] if check["name"] == "goal.folded"), None)
        if check is None or check["passed"] is None:
            result["goal_inconclusive"] += 1
        else:
            result["goal_correct" if check["passed"] == truth else "goal_wrong"] += 1
    decided = result["goal_correct"] + result["goal_wrong"]
    evaluated = decided + result["goal_inconclusive"]
    result["goal_accuracy_on_decided"] = result["goal_correct"] / decided if decided else None
    result["goal_decision_coverage"] = decided / evaluated if evaluated else None
    result["note"] = "Training diagnostics are not held-out accuracy. Inconclusive and missing cases are reported separately."
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate TrueTake goal checks against human labels.")
    parser.add_argument("batch")
    parser.add_argument("labels")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args(argv)
    try:
        result = evaluate_cards(args.batch, args.labels, args.split)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"truetake evaluate: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
