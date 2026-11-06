import json

from truetake.evaluate import evaluate_cards


def test_split_filtering_unknowns_and_denominators(tmp_path):
    cards = {clip: {"checks": [{"name": "goal.folded", "passed": value}]} for clip, value in
             (("a", True), ("b", None), ("c", False), ("test_clip", True))}
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"cards": cards}))
    labels = tmp_path / "labels.csv"
    labels.write_text("clip_id,split,goal_met\na,train,yes\nb,train,yes\nc,train,yes\nmissing,train,no\ntest_clip,test,yes\n")
    result = evaluate_cards(str(batch), str(labels), "train")
    assert result["labeled_clips"] == 4
    assert result["goal_correct"] == 1
    assert result["goal_wrong"] == 1
    assert result["goal_inconclusive"] == 1
    assert result["missing_cards"] == ["missing"]
    assert result["goal_accuracy_on_decided"] == 0.5
    assert result["goal_decision_coverage"] == 2 / 3
