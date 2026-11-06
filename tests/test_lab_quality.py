from copy import deepcopy
from pathlib import Path

import pytest

from truetake.contracts import load_contract
from truetake.quality import check_quality
from truetake.detect_claude import validate_contract_review
from test_quality import sample_table


def lab_table(seconds=15):
    table = sample_table().iloc[:seconds * 10].copy()
    contract = load_contract(str(Path(__file__).resolve().parents[1] / "contracts/mercor_folding.yaml"))
    table.at[0, "contract"] = contract
    table.at[0, "claude_review"] = {
        "evidence": [{"id": "e1", "frame_id": 0, "t": 0., "observation": "fold starts"},
                     {"id": "e2", "frame_id": len(table)-1, "t": float(table.t.iloc[-1]), "observation": "fold complete"}],
        "goal": {"verdict": "yes", "reason": "fold complete", "evidence_ids": ["e2"]},
        "task_action_shown": {"verdict": "yes", "reason": "folding shown", "evidence_ids": ["e1"]},
        "completion_required": True, "completion_override_reason": "",
        "rules": [{"rule_id": "folding_action_visible", "verdict": "yes", "reason": "task understandable", "evidence_ids": ["e1", "e2"]},
                  {"rule_id": "both_hands_visible", "verdict": "no", "reason": "brief hand exits", "evidence_ids": ["e1"]}],
        "steps": [{"name": "smooth", "evidence_ids": ["e1"]}],
        "intervals": [], "follow_up_requests": [], "provenance": {"api_calls": 1}}
    return table


def test_short_complete_clip_and_optional_hand_exits_are_accepted():
    table = lab_table(9)
    table["hands_inside"] = 0.
    checks, quality = check_quality(table)
    assert quality["status"] == "ACCEPT"
    assert quality["usable_seconds"] == pytest.approx(9)


def test_face_flags_blur_instead_of_rejecting():
    table = lab_table()
    table.at[10, "face_confidence"] = 0.95
    checks, quality = check_quality(table)
    assert quality["status"] == "ACCEPT"
    assert quality["privacy_blur_intervals"] == [(0.5, 1.7)]


def test_completion_and_task_action_are_required():
    table = lab_table()
    review = deepcopy(table.iloc[0].claude_review)
    review["goal"]["verdict"] = "no"
    table.at[0, "claude_review"] = review
    assert check_quality(table)[1]["status"] == "REJECT"
    contract = deepcopy(table.iloc[0].contract)
    contract["operator_request"] = "Partial demonstrations are allowed."
    table.at[0, "contract"] = contract
    review["completion_required"] = False
    review["completion_override_reason"] = "Operator allows partial demonstrations"
    assert check_quality(table)[1]["status"] == "ACCEPT"
    review["task_action_shown"]["verdict"] = "no"
    assert check_quality(table)[1]["status"] == "REJECT"


def test_allowed_gap_and_uncertain_long_gap():
    table = lab_table()
    review = table.iloc[0].claude_review
    review["rules"][0]["verdict"] = "no"
    interval = {"start_s": 3., "end_s": 4.5, "rule_id": "folding_action_visible", "confirmed": False,
                "action": "exclude_interval", "evidence_ids": ["e1"]}
    review["intervals"] = [interval]
    assert check_quality(table)[1]["status"] == "ACCEPT"
    interval["end_s"] = 7.
    quality = check_quality(table)[1]
    assert quality["status"] == "REVIEW"
    assert quality["excluded_intervals"] == []


def test_confirmed_dense_gap_cuts_less_and_protects_goal():
    table = lab_table()
    review = table.iloc[0].claude_review
    frames = []
    for index in range(40, 101, 5):
        review["evidence"].append({"id": f"gap{index}", "frame_id": index, "t": index / 10, "observation": "object occluded"})
        frames.append(index)
    review["provenance"] = {"api_calls": 2, "followup_frame_ids": frames}
    review["rules"][0]["verdict"] = "no"
    review["intervals"] = [{"start_s": 4., "end_s": 10., "rule_id": "folding_action_visible", "confirmed": True,
                            "action": "exclude_interval", "evidence_ids": [f"gap{index}" for index in frames]}]
    quality = check_quality(table)[1]
    assert quality["excluded_intervals"] == [(5., 9.)]
    assert quality["usable_seconds"] == pytest.approx(11)
    assert quality["status"] == "ACCEPT"
    review["goal"]["evidence_ids"] = ["gap70"]
    assert check_quality(table)[1]["excluded_intervals"] == []
    review["goal"]["evidence_ids"] = ["e2"]
    table.iloc[0].contract["duration"]["exclude_flagged_intervals"] = False
    assert check_quality(table)[1]["excluded_intervals"] == []


def test_response_validation_and_override_guard():
    table = lab_table()
    frames = {int(i): float(row.t) for i, row in table.iterrows()}
    review = deepcopy(table.iloc[0].claude_review)
    validate_contract_review(review, frames, table.iloc[0].contract, 15)
    review["rules"][1]["evidence_ids"] = []
    validate_contract_review(review, frames, table.iloc[0].contract, 15)
    assert review["rules"][1]["verdict"] == "unsure"
    review["rules"].append({"rule_id": "faces", "verdict": "yes", "reason": "model observation", "evidence_ids": []})
    validate_contract_review(review, frames, table.iloc[0].contract, 15)
    assert review["requirement_observations"][0]["rule_id"] == "faces"
    review["rules"].append({"rule_id": "invented_rule", "verdict": "yes", "reason": "unsupported", "evidence_ids": ["e1"]})
    with pytest.raises(ValueError, match="Unknown contract rule"):
        validate_contract_review(review, frames, table.iloc[0].contract, 15)
    review["rules"].pop()
    review["completion_required"] = False
    with pytest.raises(ValueError, match="override"):
        validate_contract_review(review, frames, table.iloc[0].contract, 15)


def test_uncertain_completion_never_passes_despite_high_partial_score():
    table = lab_table()
    table.iloc[0].claude_review["goal"]["verdict"] = "unsure"
    quality = check_quality(table)[1]
    assert quality["status"] == "REVIEW"
    assert quality["quality_score"] == 100
    assert quality["component_coverage"] == pytest.approx(2 / 3)
    assert quality["usable_seconds"] == 0
    assert quality["candidate_task_seconds"] == 15
    assert quality["review_items"][0]["check"] == "goal.folded"


def test_missing_review_and_required_sensors_have_explicit_reasons():
    table = lab_table()
    table.at[0, "claude_review"] = None
    assert check_quality(table)[1]["status"] == "REVIEW"
    table = lab_table()
    table["sensor_present"] = False
    assert check_quality(table)[1]["status"] == "ACCEPT"
    table.iloc[0].contract["requirements"]["sensors"]["required"] = True
    quality = check_quality(table)[1]
    assert quality["status"] == "REVIEW"
    assert any("motion data" in reason for reason in quality["decision_reasons"])


def test_unrelated_task_has_zero_usable_seconds():
    table = lab_table()
    table.iloc[0].claude_review["goal"]["verdict"] = "no"
    table.iloc[0].claude_review["task_action_shown"]["verdict"] = "no"
    quality = check_quality(table)[1]
    assert quality["status"] == "REJECT"
    assert quality["candidate_task_seconds"] == 0
    assert quality["usable_seconds"] == 0
    assert quality["submitted_seconds"] == 15
