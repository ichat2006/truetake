import numpy as np
import pandas as pd
import pytest

from truetake.edit_plan import build_edit_plan, retime_sensors
from truetake.models import Recording


def recording():
    t = np.arange(-10, 411) / 100
    imu = pd.DataFrame({"t": t, "gx": t, "gy": 2*t, "gz": -t, "ax": 0., "ay": 0., "az": 9.81})
    return Recording("sample", "unused.mp4", np.arange(40)/10,
                     {"duration_s": 4., "display_width": 160, "display_height": 96, "rotation": 0}, imu)


def export_card(rec, status="ACCEPT", cuts=None):
    return {"clip_id": rec.clip_id, "duration_s": rec.meta["duration_s"], "status": status,"license":"own",
            "completion_required": False, "excluded_intervals": cuts or [], "privacy_blur_intervals": [],
            "checks": [{"name": "process.excluded_time", "passed": True, "score": 1., "reason": "synthetic compiled exclusion", "flags": []}],
            "claude_review": {"evidence": [], "goal": {"evidence_ids": []}}}


def test_inward_snapping_and_overlapping_cut_union():
    rec = recording()
    plan = build_edit_plan(rec, export_card(rec, cuts=[[1.02, 2.08]]))
    cut = plan["applied_cuts"][0]
    assert cut["start_s"] == 1.1 and cut["end_s"] == 2.
    assert plan["output_duration_s"] == pytest.approx(3.1)
    assert len(plan["source_frame_indices"]) == 31
    assert plan["kept_segments"][1]["output_start_s"] == 1.1
    union = build_edit_plan(rec, export_card(rec, cuts=[[1., 2.], [1.5, 2.5]]))
    assert len(union["applied_cuts"]) == 1 and union["output_duration_s"] == 2.5


def test_pending_or_uncompiled_cuts_retain_all_context():
    rec = recording()
    card = export_card(rec, "REVIEW", [[1., 2.]])
    plan = build_edit_plan(rec, card)
    assert not plan["eligible_for_training"] and plan["applied_cuts"] == []
    assert plan["output_duration_s"] == 4.
    card["status"], card["checks"] = "ACCEPT", []
    assert build_edit_plan(rec, card)["applied_cuts"] == []


def test_completion_evidence_and_its_context_are_protected():
    rec = recording()
    card = export_card(rec, cuts=[[1., 2.]])
    card["completion_required"] = True
    assert build_edit_plan(rec, card)["applied_cuts"] == []
    card["claude_review"] = {"evidence": [{"id": "goal", "t": 2.5}], "goal": {"evidence_ids": ["goal"]}}
    assert build_edit_plan(rec, card)["applied_cuts"] == []
    card["claude_review"]["evidence"][0]["t"] = 3.9
    assert len(build_edit_plan(rec, card)["applied_cuts"]) == 1

def test_pending_local_cut_keeps_unverified_goal_pending():
    rec=recording()
    card=export_card(rec,'REVIEW',[[1.,2.]])
    card['completion_required']=True
    card['checks'].append({'name':'process.local_unusable_time','passed':True,
                           'reason':'exact black frames','flags':[[.5,2.5,'black frames']]})
    plan=build_edit_plan(rec,card)
    assert not plan['eligible_for_training']
    assert plan['output_duration_s']==3.
    assert len(plan['applied_cuts'])==1
    # An unrelated semantic cut cannot inherit local-detector authorization.
    card['excluded_intervals']=[[2.6,3.]]
    assert build_edit_plan(rec,card)['applied_cuts']==[]

def test_pending_local_cut_still_protects_known_completion_evidence():
    rec=recording()
    card=export_card(rec,'REVIEW',[[1.,2.]])
    card['checks'].append({'name':'process.local_unusable_time','passed':True,
                           'reason':'repeated frames','flags':[[.5,2.5,'freeze']]})
    card['claude_review']={'evidence':[{'id':'end','t':1.5}], 'goal':{'evidence_ids':['end']}}
    assert build_edit_plan(rec,card)['applied_cuts']==[]


@pytest.mark.parametrize("cuts", [[[1., 1.]], [[-1., 1.]], [[0., 5.]], [[0., float("nan")]], [[0., 4.]]])
def test_invalid_or_empty_video_plans_are_rejected(cuts):
    rec = recording()
    with pytest.raises(ValueError):
        build_edit_plan(rec, export_card(rec, cuts=cuts))


def test_sensor_values_and_original_timestamps_are_preserved_across_join():
    rec = recording()
    plan = build_edit_plan(rec, export_card(rec, cuts=[[1., 2.]]))
    sensors = retime_sensors(rec.imu, plan["kept_segments"])
    assert len(sensors) == 300 and np.all(np.diff(sensors.t) > 0)
    assert not ((sensors.source_t >= 1) & (sensors.source_t < 2)).any()
    np.testing.assert_allclose(sensors.gx, sensors.source_t)
    np.testing.assert_allclose(sensors.gy, sensors.source_t * 2)
    np.testing.assert_allclose(sensors.t, sensors.source_t - sensors.segment_id)
    assert retime_sensors(None, plan["kept_segments"]) is None


def test_face_regions_and_missing_boxes_route_to_separate_pending_copy():
    rec = recording()
    card = export_card(rec, cuts=[[1., 2.]])
    card["privacy_blur_intervals"] = [[0., 1.]]
    fallback = build_edit_plan(rec, card)
    assert not fallback["eligible_for_training"] and fallback["applied_cuts"] == []
    table = pd.DataFrame({"t": [.5], "face_boxes": [[[.7, .1, .1, .2]]]})
    plan = build_edit_plan(rec, card, table)
    assert plan["eligible_for_training"]
    region = plan["privacy_regions"][0]
    assert not region["full_frame_fallback"]
    assert region["width"] < rec.meta["display_width"]
