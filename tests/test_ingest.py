import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from truetake.ingest import load_recording, load_sensors


@pytest.fixture
def video_probe(tmp_path, monkeypatch):
    video = tmp_path / "example.mp4"
    video.touch()
    probe = {
        "streams": [{"width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"}],
        "frames": [
            {"best_effort_timestamp_time": "2.0"},
            {"best_effort_timestamp_time": "2.04"},
            {"best_effort_timestamp_time": "2.11", "pkt_duration_time": "0.03"},
        ],
    }

    def run(command, **kwargs):
        assert command[0] == "ffprobe"
        assert command[-1] == str(video.resolve())
        assert kwargs["check"] is True
        return SimpleNamespace(stdout=json.dumps(probe))

    monkeypatch.setattr("truetake.ingest.subprocess.run", run)
    return video, probe


def test_actual_frame_times_and_missing_sensors(video_probe):
    video, _ = video_probe
    rec = load_recording(str(video))
    np.testing.assert_allclose(rec.frame_times, [0, 0.04, 0.11])
    assert rec.frame_times.dtype == np.float64
    assert rec.meta["duration_s"] == pytest.approx(0.14)
    assert rec.meta["fps"] == pytest.approx(29.97003)
    assert rec.meta["duration_estimated"] is False
    assert rec.imu is None
    assert rec.clip_id == "example"


@pytest.mark.parametrize("angle", [90, -90, 180, 270])
def test_rotation_changes_display_dimensions_only(video_probe, angle):
    video, probe = video_probe
    probe["streams"][0]["side_data_list"] = [{"rotation": angle}]
    rec = load_recording(str(video))
    assert rec.meta["rotation"] == angle % 360
    assert rec.meta["width"] == 1920
    assert rec.meta["display_width"] == (1080 if angle % 180 else 1920)
    assert rec.meta["display_height"] == (1920 if angle % 180 else 1080)


def test_last_frame_duration_estimate_is_explicit(video_probe):
    video, probe = video_probe
    del probe["frames"][-1]["pkt_duration_time"]
    rec = load_recording(str(video))
    assert rec.meta["duration_estimated"] is True
    assert rec.meta["duration_s"] == pytest.approx(0.165)


@pytest.mark.parametrize("timestamp", ["nan", "inf", "1.0", "2.0", "bad"])
def test_bad_frame_timestamps_are_rejected(video_probe, timestamp):
    video, probe = video_probe
    probe["frames"][1]["best_effort_timestamp_time"] = timestamp
    with pytest.raises(ValueError, match="timestamp"):
        load_recording(str(video))


def test_missing_timestamp_is_not_replaced_by_nominal_fps(video_probe):
    video, probe = video_probe
    del probe["frames"][1]["best_effort_timestamp_time"]
    with pytest.raises(ValueError, match="presentation timestamp"):
        load_recording(str(video))


def test_no_video_frames(video_probe):
    video, probe = video_probe
    probe["frames"] = []
    with pytest.raises(ValueError, match="no decodable frames"):
        load_recording(str(video))


@pytest.mark.parametrize("rate", ["0/0", "bad", "-30/1"])
def test_invalid_fps(video_probe, rate):
    video, probe = video_probe
    probe["streams"][0]["avg_frame_rate"] = rate
    with pytest.raises(ValueError, match="frame rate"):
        load_recording(str(video))


@pytest.mark.parametrize(
    "error,message",
    [
        (FileNotFoundError(), "ffprobe is required"),
        (subprocess.TimeoutExpired("ffprobe", 120), "timed out"),
        (subprocess.CalledProcessError(1, "ffprobe"), "could not decode"),
    ],
)
def test_probe_errors_are_actionable(video_probe, monkeypatch, error, message):
    video, _ = video_probe

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("truetake.ingest.subprocess.run", fail)
    with pytest.raises((ValueError, RuntimeError), match=message):
        load_recording(str(video))


def test_normalized_sensors_preserve_origin_and_units(tmp_path, video_probe):
    sensors = tmp_path / "sensors.csv"
    sensors.write_text(
        "t,gx,gy,gz,ax,ay,az,extra\n"
        "-0.1,1.5,-2,0,0,0,9.80665,unused\n"
        "0.1,2.5,-3,0,0,0,9.80665,unused\n"
    )
    rec = load_recording(str(video_probe[0]), str(sensors))
    assert list(rec.imu.columns) == ["t", "gx", "gy", "gz", "ax", "ay", "az"]
    assert all(dtype == np.float64 for dtype in rec.imu.dtypes)
    assert rec.imu["t"].tolist() == [-0.1, 0.1]
    assert rec.imu["gx"].tolist() == [1.5, 2.5]
    assert rec.imu["az"].iloc[0] == 9.80665


@pytest.mark.parametrize(
    "body,message",
    [
        ("", "no samples"),
        ("0,1,2,3,4,5,\n", "finite"),
        ("0,1,2,inf,4,5,6\n", "finite"),
        ("0,bad,2,3,4,5,6\n", "only numbers"),
        ("0,1,2,3,4,5,6\n0,1,2,3,4,5,6\n", "strictly increasing"),
        ("1,1,2,3,4,5,6\n0,1,2,3,4,5,6\n", "strictly increasing"),
    ],
)
def test_invalid_sensor_samples(tmp_path, body, message):
    path = tmp_path / "sensors.csv"
    path.write_text("t,gx,gy,gz,ax,ay,az\n" + body)
    with pytest.raises(ValueError, match=message):
        load_sensors(str(path))


def test_sensor_columns_are_required(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("timestamp,gyro_x\n0,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_sensors(str(path))


def test_supplied_missing_sensor_file_is_not_treated_as_no_sensors(video_probe, tmp_path):
    with pytest.raises(FileNotFoundError, match="Sensor file not found"):
        load_recording(str(video_probe[0]), str(tmp_path / "absent.csv"))


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        load_recording(str(tmp_path / "absent.mp4"))


def test_wrong_file_types_are_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        load_recording(str(tmp_path / "not-video.txt"))
    with pytest.raises(ValueError, match="normalized .csv"):
        load_sensors(str(tmp_path / "not-sensors.txt"))
