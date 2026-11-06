"""Exercise real ffmpeg/ffprobe decoding and the CLI without touching data/."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from truetake.ingest import load_recording


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Video integration tests require ffmpeg and ffprobe",
)


@pytest.fixture(params=["constant", "variable"])
def generated_clip(request, tmp_path):
    video = tmp_path / f"{request.param}.mp4"
    command = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x180:rate=30:duration=1",
    ]
    if request.param == "variable":
        command += [
            "-vf", "select='eq(n,0)+eq(n,3)+eq(n,9)+eq(n,18)'",
            "-fps_mode", "vfr",
        ]
    command += ["-c:v", "mpeg4", "-threads", "1", str(video)]
    subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    return video, request.param


def test_real_video_timing(generated_clip):
    video, kind = generated_clip
    rec = load_recording(str(video))
    assert rec.imu is None
    assert rec.meta["width"] == 320
    assert rec.meta["height"] == 180
    if kind == "constant":
        assert len(rec.frame_times) == 30
        np.testing.assert_allclose(rec.frame_times, np.arange(30) / 30, atol=1e-6)
        assert rec.meta["duration_s"] == pytest.approx(1.0, abs=1e-5)
    else:
        np.testing.assert_allclose(rec.frame_times, [0.0, 0.1, 0.3, 0.6], atol=1e-6)


def test_documented_cli_with_video_sensors_and_contract(generated_clip, tmp_path):
    video, _ = generated_clip
    sensors = tmp_path / "sensors.csv"
    samples = np.zeros((101, 7))
    samples[:, 0] = np.arange(101) / 100
    samples[:, 6] = 9.80665
    np.savetxt(sensors, samples, delimiter=",", header="t,gx,gy,gz,ax,ay,az", comments="")
    root = Path(__file__).resolve().parents[1]
    contract = root / "contracts" / "fold.yaml"
    result = subprocess.run(
        [sys.executable, str(root / "cli.py"), str(video),
         "--sensors", str(sensors), "--contract", str(contract),
         "--diagnose-gyro", "--output-dir", str(tmp_path / "results")],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30,
    )
    summary = json.loads(result.stdout)
    assert summary["clip_id"] == video.stem
    assert summary["sensor_samples"] == 101
    assert summary["sensor_rate_hz"] == pytest.approx(100.0)
    assert summary["sensor_status"] == "loaded"
    assert summary["task"] == "fold a T-shirt"
    assert summary["review_status"] == "not_run"
    assert summary["stage"] == "verify"
    assert summary["verification"]["passed"] is None
    assert summary["gyro_alignment"]["status"] == "insufficient_overlap"
    diagnostic_paths = summary["gyro_diagnostics"]
    assert diagnostic_paths["candidate_profile"] is None
    assert Path(diagnostic_paths["plot"]).read_bytes().startswith(b"\x89PNG")
    assert json.loads(Path(diagnostic_paths["report"]).read_text())["status"] == "insufficient_overlap"
    records = json.loads(Path(summary["per_frame_table"]).read_text())
    assert len(records) == (10 if generated_clip[1] == "constant" else 4)
    assert all(row["brightness"] is not None and row["blur_var"] is not None for row in records)
    assert records[0]["frame_change"] is None
    assert all(row["imu_valid"] for row in records)
    assert records[0]["flow_status"] == "no_previous"
    assert records[0]["flow_pan"] is None
    assert records[0]["gyro_alignment"] == summary["gyro_alignment"]
    assert all(row["gyro_alignment"] is None for row in records[1:])
    assert summary["flow_valid_pairs"] >= 1
    for row in records[1:]:
        if row["flow_status"] == "ok":
            assert row["flow_pan"] is not None
            assert row["flow_inlier_fraction"] >= 0.5
        else:
            assert row["flow_pan"] is None
            assert row["flow_roll"] is None
    if generated_clip[1] == "constant":
        # Animated testsrc objects need not admit one camera-motion transform.
        assert all(row["flow_dt"] == pytest.approx(0.1) for row in records[1:])
    else:
        assert records[-1]["flow_status"] == "frame_gap"
        assert records[-1]["flow_pan"] is None
    assert not result.stderr


def test_corrupt_video_is_rejected(tmp_path):
    video = tmp_path / "corrupt.mp4"
    video.write_bytes(b"This is not a video")
    with pytest.raises(ValueError, match="could not decode"):
        load_recording(str(video))


def test_rerun_recording_is_readable_without_a_viewer(generated_clip, tmp_path):
    from truetake.frame_table import build_table
    from truetake.detect_basic import detect_basic
    from truetake.viz_rerun import save_recording

    rec = load_recording(str(generated_clip[0]))
    table = detect_basic(rec, build_table(rec))
    path = tmp_path / "timeline.rrd"
    save_recording(rec, table, {"status": "REVIEW", "checks": []}, str(path))
    assert path.stat().st_size > 1000
    subprocess.run([shutil.which("rerun"), "rrd", "verify", str(path)],
                   capture_output=True, check=True, timeout=30)


def test_selected_evidence_uses_real_frames_and_preserves_source(generated_clip, tmp_path):
    import cv2
    import pandas as pd
    from truetake.frame_table import build_table
    from truetake.evidence import select_evidence, save_selected_frames
    source = generated_clip[0]
    before = source.read_bytes()
    rec = load_recording(str(source))
    table = select_evidence(build_table(rec))
    table["thumbnail_path"] = pd.Series([None] * len(table), dtype=object)
    result = save_selected_frames(rec, table, str(tmp_path / "evidence"))
    assert result.thumbnail_path.notna().sum() == min(8, len(table))
    for _, row in result[result.claude_selected].iterrows():
        assert cv2.imread(row.thumbnail_path).shape == (180, 320, 3)
        assert row.t == pytest.approx(rec.frame_times[row.source_frame])
    assert source.read_bytes() == before


def test_browser_preview_preserves_timing_and_is_cached(generated_clip, tmp_path):
    from truetake.preview import browser_preview

    source = generated_clip[0]
    before = source.read_bytes()
    path = Path(browser_preview(str(source), str(tmp_path / "previews")))
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    np.testing.assert_allclose(load_recording(str(path)).frame_times,
                               load_recording(str(source)).frame_times, atol=1e-5)
    modified = path.stat().st_mtime_ns
    assert browser_preview(str(source), str(tmp_path / "previews")) == str(path)
    assert path.stat().st_mtime_ns == modified
    assert source.read_bytes() == before
