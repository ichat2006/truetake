import numpy as np
import pandas as pd
import pytest

from truetake.verify_gyro import measure_alignment, check_verification, _interpolate


MAPPING = {"flow_pan": {"axis": "gyro_gy", "sign": -1},
           "flow_tilt": {"axis": "gyro_gz", "sign": 1},
           "flow_roll": {"axis": "gyro_gx", "sign": 1}}


def signals(t):
    return (np.sin(2 * np.pi * (0.23 * t + 0.004 * t**2)) + 0.4 * np.cos(2 * np.pi * 0.71 * t),
            np.sin(2 * np.pi * 0.37 * t) + 0.3 * np.sin(2 * np.pi * 0.83 * t),
            np.cos(2 * np.pi * 0.51 * t) + 0.5 * np.sin(2 * np.pi * 0.17 * t))


def fixture_table(offset=0.7, calibrated=False):
    t = np.arange(0, 20, 0.1)
    flow_t = t + 0.05
    v = signals(flow_t)
    g = signals(t - offset)
    table = pd.DataFrame({"t": t, "sample_end": t + 0.1, "sensor_present": True,
                          "flow_t": flow_t, "flow_status": "ok",
                          "flow_pan": v[0], "flow_tilt": v[1], "flow_roll": v[2],
                          "gyro_gx": g[2], "gyro_gy": -g[0], "gyro_gz": g[1]})
    table["clip_meta"] = pd.Series([{}] + [None] * (len(t) - 1), dtype=object)
    if calibrated:
        table.at[0, "clip_meta"] = {"gyro_calibration": {"status": "calibrated", "axis_mapping": MAPPING,
                                                       "min_match_score": 0.9, "train_clip_ids": ["synthetic_train"]}}
    return table


@pytest.mark.parametrize("offset", [-0.8, 0.0, 0.7, 1.3])
def test_estimates_signed_offset_and_permutation(offset):
    table = measure_alignment(fixture_table(offset))
    result = table.iloc[0].gyro_alignment
    assert result["status"] == "measured"
    assert result["offset_s"] == pytest.approx(offset, abs=0.1)
    assert result["score"] > 0.95
    assert result["axis_mapping"] == MAPPING
    assert result["lag_consistent"]
    check = check_verification(table)[0]
    assert check.passed is None
    assert "uncalibrated" in check.reason
    assert bool(check.flags) == (abs(offset) > 1)


def test_fixed_calibration_can_pass_without_relearning_mapping():
    table = measure_alignment(fixture_table(calibrated=True))
    assert check_verification(table)[0].passed is True


def test_fixed_mapping_does_not_flip_signs_to_hide_mismatch():
    table = fixture_table(calibrated=True)
    for axis in ("gyro_gx", "gyro_gy", "gyro_gz"):
        table[axis] *= -1
    measured = measure_alignment(table)
    assert measured.iloc[0].gyro_alignment["axis_mapping"] == MAPPING
    assert check_verification(measured)[0].passed is not True


def test_no_sensors_exact_reason():
    table = fixture_table()
    table["sensor_present"] = False
    measured = measure_alignment(table)
    check = check_verification(measured)[0]
    assert check.passed is None
    assert check.reason == "no motion data"
    assert measured.iloc[0].gyro_alignment["offset_s"] is None


def test_flat_gyro_is_inconclusive():
    table = fixture_table(calibrated=True)
    table[["gyro_gx", "gyro_gy", "gyro_gz"]] = 0.
    measured = measure_alignment(table)
    assert measured.iloc[0].gyro_alignment["score"] is None
    assert check_verification(measured)[0].passed is None


def test_missing_flow_is_not_interpreted_as_zero_motion():
    table = fixture_table(calibrated=True)
    table["flow_status"] = "insufficient_features"
    measured = measure_alignment(table)
    assert measured.iloc[0].gyro_alignment["score"] is None
    assert check_verification(measured)[0].passed is None


def test_short_overlap_is_inconclusive():
    table = measure_alignment(fixture_table().iloc[:30])
    assert table.iloc[0].gyro_alignment["status"] == "insufficient_overlap"


def test_interpolation_keeps_missing_samples_and_long_gaps():
    result = _interpolate(np.array([0., 0.1, 0.2, 0.5]),
                          np.array([1., np.nan, 3., 4.]),
                          np.array([-0.1, 0., 0.05, 0.1, 0.15, 0.35, 0.5, 0.6]))
    assert result[1] == 1
    assert result[6] == 4
    assert np.isnan(result[[0, 2, 3, 4, 5, 7]]).all()


def test_grid_roundoff_does_not_blend_sample_into_missing_neighbor():
    result = _interpolate(np.array([0., 0.1, 0.2]), np.array([1., 2., np.nan]),
                          np.array([0.1 + 1e-15, 0.2 - 1e-15]))
    assert result[0] == 2
    assert np.isnan(result[1])


def test_search_boundary_never_passes():
    table = measure_alignment(fixture_table(offset=2.0, calibrated=True))
    assert table.iloc[0].gyro_alignment["at_search_boundary"]
    assert check_verification(table)[0].passed is None


def test_mapping_must_use_each_sensor_axis_once():
    table = fixture_table(calibrated=True)
    bad_mapping = {key: {"axis": "gyro_gx", "sign": 1} for key in MAPPING}
    table.at[0, "clip_meta"] = {"gyro_calibration": {"axis_mapping": bad_mapping}}
    with pytest.raises(ValueError, match="signed permutation"):
        measure_alignment(table)


def test_axes_disagreeing_about_offset_are_inconclusive():
    table = fixture_table(calibrated=True)
    t = table.t.to_numpy()
    table["gyro_gy"] = -signals(t + 0.8)[0]
    table["gyro_gz"] = signals(t - 0.7)[1]
    table["gyro_gx"] = signals(t - 1.5)[2]
    measured = measure_alignment(table)
    assert not measured.iloc[0].gyro_alignment["lag_consistent"]
    assert check_verification(measured)[0].passed is False


def test_low_score_rejects_even_when_axis_lags_disagree():
    table = measure_alignment(fixture_table(calibrated=True))
    table.at[0, "gyro_alignment"]["score"] = 0.1
    table.at[0, "gyro_alignment"]["lag_consistent"] = False
    check = check_verification(table)[0]
    assert check.passed is False
    assert check.reason == "phone motion doesn't match the video"


def test_gyro_review_zone_comes_from_profile_without_overriding_failure():
    table = fixture_table(calibrated=True)
    table.at[0, "clip_meta"]["gyro_calibration"].update(review_low=0.95, review_high=1.)
    table = measure_alignment(table)
    assert table.iloc[0].gyro_alignment["close_call"]


def test_two_strong_axes_can_agree_despite_a_weak_stabilized_axis():
    table = fixture_table(calibrated=True)
    rng = np.random.default_rng(37)
    table["flow_tilt"] = rng.normal(size=len(table))
    table.at[0, "clip_meta"]["gyro_calibration"]["min_match_score"] = 0.5
    measured = measure_alignment(table)
    result = measured.iloc[0].gyro_alignment
    assert result["lag_consistent"]
    assert abs(result["axis_offsets_s"]["flow_tilt"] - result["offset_s"]) > 0.2
    assert check_verification(measured)[0].passed is True
