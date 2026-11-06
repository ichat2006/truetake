import numpy as np
import pandas as pd

from truetake.quality import check_quality, intervals


def sample_table(goal="yes"):
    times = np.arange(150) / 10
    table = pd.DataFrame({"t": times, "sample_end": times + 0.1, "sensor_present": False,
        "people_status": "ok", "hand_count": 2., "hands_inside": 1., "hand_motion": 0.1,
        "face_confidence": 0., "frame_change": 0.1, "brightness": 120., "blur_var": 150.,
        "clip_status": "ok", "clothing_similarity": 0.3, "no_clothing_similarity": 0.1})
    for name in ("clip_meta", "contract", "claude_review"):
        table[name] = pd.Series([None] * len(table), dtype=object)
    table.at[0, "clip_meta"] = {"display_width": 1920, "display_height": 1080, "fps": 29.97,
                                "garment_threshold_calibrated": True}
    table.at[0, "contract"] = {"requirements": {"hard": ["landscape", "min_1080p", "min_30fps", "no_faces"]}, "min_usable_seconds": 10}
    table.at[0, "claude_review"] = {"goal": {"verdict": goal, "reason": "fixture evidence"}}
    return table


def test_good_evidence_accepts_and_submitted_time_not_double_counted():
    checks, quality = check_quality(sample_table())
    assert quality["status"] == "ACCEPT"
    assert quality["usable_seconds"] > 14.9
    assert quality["quality_score"] == 100
    assert all(check.passed is True for check in checks)


def test_unknown_goal_is_review_and_no_goal_is_reject():
    assert check_quality(sample_table("unsure"))[1]["status"] == "REVIEW"
    assert check_quality(sample_table("no"))[1]["status"] == "REJECT"


def test_process_gap_trims_time_instead_of_rejecting_whole_clip():
    table = sample_table()
    table.loc[(table.t >= 5) & (table.t < 6), ["hands_inside", "clothing_similarity"]] = [0., 0.]
    checks, quality = check_quality(table)
    assert 13.9 < quality["usable_seconds"] < 14.1
    assert quality["status"] == "ACCEPT"
    assert len(quality["usable_intervals"]) == 2


def test_face_rejection_and_flags():
    table = sample_table()
    table.at[10, "face_confidence"] = 0.95
    checks, quality = check_quality(table)
    assert quality["status"] == "REJECT"
    check = next(check for check in checks if check.name == "requirements.no_faces")
    assert check.flags[0][0] == 1.0


def test_missing_detector_does_not_create_usable_time():
    table = sample_table()
    table["clip_status"] = "not_sampled"
    checks, quality = check_quality(table)
    assert quality["status"] == "REVIEW"
    assert quality["usable_seconds"] == 0


def test_uncalibrated_clothing_model_cannot_reject_for_low_usable_time():
    table = sample_table()
    table.at[0, "clip_meta"] = {**table.iloc[0].clip_meta, "garment_threshold_calibrated": False}
    table["clothing_similarity"] = 0.
    checks, quality = check_quality(table)
    assert quality["status"] == "REVIEW"
    assert next(check for check in checks if check.name == "process.garment_visible").passed is None
    assert quality["usable_seconds"] == 0


def test_large_video_gap_is_not_counted_as_observed_time():
    table = pd.DataFrame({"t": [0., 10.], "sample_end": [10., 11.]})
    assert intervals(table, np.array([True, True])) == [(0., 0.1), (10., 10.1)]
