import cv2
import numpy as np
import pandas as pd
import pytest

from truetake.detect_flow import detect_flow, measure_pair


@pytest.fixture
def texture():
    image = np.random.default_rng(42).integers(0, 256, (180, 320), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


@pytest.mark.parametrize("dx,dy,dt", [(3, -2, 0.1), (-2, 1, 0.05)])
def test_known_translation_and_actual_interval(texture, dx, dy, dt):
    moved = cv2.warpAffine(texture, np.float32([[1, 0, dx], [0, 1, dy]]), (320, 180))
    result = measure_pair(texture, moved, dt)
    assert result["flow_status"] == "ok"
    assert result["flow_pan"] == pytest.approx(dx / dt, abs=0.5)
    assert result["flow_tilt"] == pytest.approx(dy / dt, abs=0.5)
    assert result["flow_roll"] == pytest.approx(0, abs=0.01)
    assert result["flow_inlier_fraction"] > 0.8


def test_known_roll_does_not_create_pan(texture):
    matrix = cv2.getRotationMatrix2D((159.5, 89.5), 1.0, 1.0)
    rotated = cv2.warpAffine(texture, matrix, (320, 180))
    result = measure_pair(texture, rotated, 0.1)
    assert result["flow_status"] == "ok"
    assert result["flow_roll"] == pytest.approx(-np.deg2rad(1) / 0.1, abs=0.01)
    assert result["flow_pan"] == pytest.approx(0, abs=0.5)
    assert result["flow_tilt"] == pytest.approx(0, abs=0.5)


def test_static_textured_frame_is_measurable_zero(texture):
    result = measure_pair(texture, texture, 0.1)
    assert result["flow_status"] == "ok"
    assert result["flow_pan"] == pytest.approx(0, abs=0.01)
    assert result["flow_tilt"] == pytest.approx(0, abs=0.01)


def test_featureless_frames_are_unknown_not_zero_motion():
    image = np.zeros((180, 320), dtype=np.uint8)
    result = measure_pair(image, image, 0.1)
    assert result["flow_status"] == "insufficient_features"
    assert result["flow_pan"] is None
    assert result["flow_roll"] is None


def test_large_frame_gap_is_not_bridged(texture):
    result = measure_pair(texture, texture, 0.4)
    assert result["flow_status"] == "frame_gap"
    assert result["flow_pan"] is None


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_invalid_interval(texture, dt):
    with pytest.raises(ValueError, match="interval"):
        measure_pair(texture, texture, dt)


def test_rejected_fit_has_no_rates(texture, monkeypatch):
    monkeypatch.setattr("truetake.detect_flow.cv2.estimateAffinePartial2D",
                        lambda source, target, **kwargs: (np.eye(2, 3), np.zeros((len(source), 1))))
    result = measure_pair(texture, texture, 0.1)
    assert result["flow_status"] == "low_inliers"
    assert result["flow_inlier_fraction"] == 0
    assert result["flow_pan"] is None


def test_table_midpoints_and_recovery_after_gap(texture, monkeypatch):
    images = [cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)] * 4
    monkeypatch.setattr("truetake.detect_flow.iter_sampled_frames", lambda *args: enumerate(images))
    original = pd.DataFrame({"t": [0., 0.1, 0.5, 0.6]})
    table = detect_flow(None, original)
    assert table.flow_status.tolist() == ["no_previous", "ok", "frame_gap", "ok"]
    assert np.isnan(table.flow_pan.iloc[0])
    assert table.flow_t.iloc[1] == pytest.approx(0.05)
    assert table.flow_t.iloc[3] == pytest.approx(0.55)
    assert table.flow_dt.iloc[3] == pytest.approx(0.1)
    assert "flow_pan" not in original
