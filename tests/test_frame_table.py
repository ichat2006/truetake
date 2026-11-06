import numpy as np
import pandas as pd
import pytest

from truetake.frame_table import build_table
from truetake.models import Recording


def recording(times, duration, imu=None):
    return Recording("sample", "sample.mp4", np.array(times, dtype=float),
                     {"duration_s": duration}, imu)


def test_sampling_30fps_preserves_frame_indices_and_total_duration():
    table = build_table(recording(np.arange(30) / 30, 1), {"task": "fold"})
    assert table.source_frame.tolist() == list(range(0, 30, 3))
    assert len(table) == 10
    assert (table.sample_end - table.t).sum() == pytest.approx(1)
    assert table.iloc[0].contract == {"task": "fold"}
    assert table.contract.iloc[1:].isna().all()
    assert not table.sensor_present.any()
    assert not table.imu_valid.any()
    assert table[["gx", "gy", "gz", "ax", "ay", "az"]].isna().all().all()


def test_vfr_does_not_duplicate_or_invent_source_frames():
    table = build_table(recording([0, 0.11, 0.45, 0.8], 1))
    assert table.source_frame.tolist() == [0, 1, 2, 3]
    assert table.t.tolist() == [0, 0.11, 0.45, 0.8]


def test_low_source_fps_is_not_upsampled():
    table = build_table(recording([0, 0.5], 1))
    assert len(table) == 2


def test_sensors_no_extrapolation_or_interpolation_through_large_gaps():
    times = np.array([0.1, 0.15, 0.2, 0.5, 0.55])
    imu = pd.DataFrame({"t": times, **{key: times * 2 for key in ("gx", "gy", "gz", "ax", "ay", "az")}})
    table = build_table(recording(np.arange(7) / 10, 0.7, imu))
    assert table.imu_valid.tolist() == [False, True, True, False, False, True, False]
    assert table.loc[1, "gx"] == pytest.approx(0.2)
    assert np.isnan(table.loc[3, "gx"])


def test_interpolates_inside_short_gap():
    imu = pd.DataFrame({"t": [0.05, 0.15], **{key: [1., 3.] for key in ("gx", "gy", "gz", "ax", "ay", "az")}})
    table = build_table(recording([0, 0.1], 0.2, imu))
    assert table.loc[1, "gx"] == pytest.approx(2)


@pytest.mark.parametrize("times,duration", [([], 1), ([0, 0], 1), ([0, float("nan")], 1), ([0.1], 1), ([0, 1], 1)])
def test_invalid_recording(times, duration):
    with pytest.raises(ValueError, match="increasing frame times"):
        build_table(recording(times, duration))


def test_gyro_is_smoothed_before_downsampling():
    t = np.arange(401) / 200
    oscillation = np.cos(2 * np.pi * 20 * t)
    imu = pd.DataFrame({"t": t, **{axis: oscillation for axis in ("gx", "gy", "gz", "ax", "ay", "az")}})
    table = build_table(recording(np.arange(60) / 30, 2, imu))
    interior = table[(table.t >= 0.5) & (table.t <= 1.5)]
    assert interior.gx.min() > 0.95  # Unfiltered sampling would alias to a constant.
    assert interior.gyro_gx.abs().max() < 0.06


def test_gyro_smoothing_does_not_cross_sensor_gap():
    t = np.array([0., 0.01, 0.5, 0.51])
    imu = pd.DataFrame({"t": t, **{axis: [0., 0., 100., 100.] for axis in ("gx", "gy", "gz", "ax", "ay", "az")}})
    table = build_table(recording(np.arange(6) / 10, 0.6, imu))
    assert table.gyro_gx.iloc[0] == 0
    assert table.gyro_gx.iloc[-1] == 100
    assert table.gyro_gx.iloc[1:-1].isna().all()
