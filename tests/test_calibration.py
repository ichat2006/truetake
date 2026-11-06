import json

import numpy as np
import pytest

from truetake.calibration import diagnose_alignment, load_profile, fit_reference
from truetake.verify_gyro import measure_alignment, check_verification
from truetake.viz_alignment import plot_alignment
from test_verify_gyro import fixture_table, MAPPING


def test_five_second_window_is_measurable_with_full_sensor_context():
    result = measure_alignment(fixture_table(), interval=(5., 10.)).iloc[0].gyro_alignment
    assert result["status"] == "measured"
    assert result["offset_s"] == pytest.approx(0.7, abs=0.100001)
    assert result["score"] > 0.95
    assert len(result["lag_curve"]) == 41


def test_later_windows_use_fixed_mapping_and_reference_offset():
    measured = diagnose_alignment(fixture_table(), "synthetic")
    result = measured.iloc[0].gyro_diagnostics
    assert result["profile"]["axis_mapping"] == MAPPING
    assert result["profile"]["status"] == "provisional"
    assert result["profile"]["min_match_score"] is None
    assert result["calibration_interval_s"] == pytest.approx([0., 20.])
    assert result["summary"]["total_windows"] == 16
    assert result["windows"][0]["start_s"] == 0.
    assert all(window["reference_score"] > 0.95 for window in result["windows"])
    assert result["summary"]["offset_agreement_windows"] == 16


def test_whole_clip_fit_works_when_first_ten_seconds_are_stationary():
    table = fixture_table()
    for axis in MAPPING:
        table.loc[table.flow_t < 10, axis] = 0.
    table.loc[table.t < 10, ["gyro_gx", "gyro_gy", "gyro_gz"]] = 0.
    result = diagnose_alignment(table, "later_motion").iloc[0].gyro_diagnostics
    assert result["profile"]["axis_mapping"] == MAPPING
    assert result["baseline"]["offset_s"] == pytest.approx(0.7, abs=0.100001)
    excluded = [w for w in result["windows"] if w["status"] == "insufficient_motion"]
    assert excluded
    assert all(w["gyro_std"] < 0.02 or w["at_search_boundary"] for w in excluded)
    assert result["summary"]["mapping_comparable_windows"] <= result["summary"]["measured_windows"]


def test_reference_does_not_use_an_unrelated_weak_axis():
    table = fixture_table()
    table["flow_tilt"] = np.random.default_rng(12).normal(size=len(table))
    baseline = fit_reference(table)
    assert "flow_tilt" not in baseline["reference_axes"]
    assert set(baseline["reference_axes"]) == {"flow_pan", "flow_roll"}
    assert baseline["offset_s"] == pytest.approx(0.7, abs=0.100001)


def test_boundary_windows_are_excluded_from_all_agreement_counts():
    result = diagnose_alignment(fixture_table(offset=2.0), "boundary").iloc[0].gyro_diagnostics
    excluded = [window for window in result["windows"] if window["at_search_boundary"]]
    assert excluded
    assert all(window["status"] == "insufficient_motion" for window in excluded)
    assert result["summary"]["mapping_comparable_windows"] <= result["summary"]["measured_windows"]
    assert result["summary"]["offset_agreement_windows"] <= result["summary"]["measured_windows"]


def test_missing_sensors_produce_no_profile_and_explanatory_plot(tmp_path):
    table = fixture_table()
    table["sensor_present"] = False
    measured = diagnose_alignment(table, "no_sensors")
    result = measured.iloc[0].gyro_diagnostics
    assert result["status"] == "no_motion_data"
    assert result["profile"] is None
    assert result["windows"] == []
    plot = tmp_path / "missing.png"
    plot_alignment(measured, str(plot))
    assert plot.read_bytes().startswith(b"\x89PNG")


def test_diagnostic_plot_and_profile_roundtrip_stay_inconclusive(tmp_path):
    table = fixture_table()
    measured = diagnose_alignment(table, "synthetic")
    result = measured.iloc[0].gyro_diagnostics
    plot = tmp_path / "alignment.png"
    plot_alignment(measured, str(plot))
    assert plot.stat().st_size > 1000
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result["profile"]))
    table.at[0, "clip_meta"] = {"gyro_calibration": load_profile(str(path))}
    verified = measure_alignment(table)
    assert not verified.iloc[0].gyro_alignment["mapping_calibrated"]
    assert check_verification(verified)[0].passed is None


@pytest.mark.parametrize("profile", [[], {}, {"schema_version": 1, "status": "calibrated"},
                                      {"schema_version": 1, "status": "provisional", "min_match_score": 0.5},
                                      {"schema_version": 1, "status": "provisional", "axis_mapping": {}}])
def test_invalid_profiles_rejected(tmp_path, profile):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_profile(str(path))


@pytest.mark.parametrize("threshold", [0., 0.6, 1.])
def test_calibrated_profile_roundtrip_can_enable_verdict(tmp_path, threshold):
    profile = {"schema_version": 1, "status": "calibrated", "axis_mapping": MAPPING,
               "min_match_score": threshold, "train_clip_ids": ["synthetic_train"]}
    path = tmp_path / "calibrated.json"
    path.write_text(json.dumps(profile))
    table = fixture_table()
    table.at[0, "clip_meta"] = {"gyro_calibration": load_profile(str(path))}
    result = check_verification(measure_alignment(table))[0]
    assert result.passed is (threshold < 1.)


@pytest.mark.parametrize("change", [{"min_match_score": -0.1}, {"min_match_score": 1.1},
                                     {"min_match_score": True}, {"min_match_score": float("nan")},
                                     {"train_clip_ids": []}, {"train_clip_ids": ["train", "train"]}])
def test_calibrated_profiles_require_valid_threshold_and_provenance(tmp_path, change):
    profile = {"schema_version": 1, "status": "calibrated", "axis_mapping": MAPPING,
               "min_match_score": 0.8, "train_clip_ids": ["train"], **change}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_profile(str(path))
