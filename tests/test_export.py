import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
import pytest

from truetake.export import export_clip, export_collection, write_collection_manifest
from truetake.ingest import load_recording, load_sensors
from test_edit_plan import export_card


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "sample.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10:duration=4",
                    "-c:v", "libx264", "-threads", "1", "-bf", "0", str(path)], capture_output=True, check=True)
    rec = load_recording(str(path))
    t = np.arange(401) / 100
    rec.imu = pd.DataFrame({"t": t, "gx": t, "gy": t*2, "gz": -t, "ax": 0., "ay": 0., "az": 9.81})
    return rec


def frame_at(path, time):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        ok, frame = capture.read()
        assert ok
        return frame
    finally:
        capture.release()


def test_trim_sensor_export_frame_map_and_validated_cache(source, tmp_path, monkeypatch):
    original = hashlib.sha256(Path(source.video_path).read_bytes()).hexdigest()
    result = export_clip(source, export_card(source, cuts=[[1., 2.]]), str(tmp_path / "exports"))
    assert result["export_status"] == "exported" and result["eligible_for_training"]
    encoded = load_recording(result["video_path"], result["sensors_path"])
    assert len(encoded.frame_times) == 30 and encoded.meta["duration_s"] == pytest.approx(3.)
    mapping = pd.read_csv(result["frame_map_path"])
    assert mapping.source_frame.tolist() == list(range(10)) + list(range(20, 40))
    np.testing.assert_allclose(mapping.output_t, np.arange(30)/10, atol=.0001)
    sensors = pd.read_csv(result["sensors_path"])
    assert len(sensors) == 300 and len(encoded.imu) == 300
    np.testing.assert_allclose(sensors.gx, sensors.source_t)
    np.testing.assert_allclose(sensors.t, sensors.source_t - sensors.segment_id)
    assert result["sensor_join_times_s"] == [1.]
    def forbidden(*args, **kwargs): raise AssertionError("matching export cache must not invoke FFmpeg")
    monkeypatch.setattr("truetake.export.subprocess.run", forbidden)
    cached = export_clip(source, export_card(source, cuts=[[1., 2.]]), str(tmp_path / "exports"))
    assert cached["cached"] and cached["video_path"] == result["video_path"]
    assert hashlib.sha256(Path(source.video_path).read_bytes()).hexdigest() == original


def test_export_is_h264_yuv420p_with_faststart(source, tmp_path):
    result = export_clip(source, export_card(source), str(tmp_path / "exports"))
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt", "-of", "json", result["video_path"]],
                           check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert result["video_encoding"]["codec_name"] == "h264"
    # MP4 header metadata must precede the media payload for progressive loading.
    import struct
    atoms = []
    with Path(result["video_path"]).open("rb") as handle:
        while header := handle.read(8):
            size, kind = struct.unpack(">I4s", header)
            atoms.append(kind)
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                handle.seek(size - 16, 1)
            elif size:
                handle.seek(size - 8, 1)
            else:
                break
    assert atoms.index(b"moov") < atoms.index(b"mdat")


def test_force_regenerates_a_matching_cached_export(source, tmp_path, monkeypatch):
    directory = str(tmp_path / "exports")
    first = export_clip(source, export_card(source), directory)
    real_run = subprocess.run
    encodes = []

    def tracked(command, *args, **kwargs):
        if command[0] == "ffmpeg":
            encodes.append(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr("truetake.export.subprocess.run", tracked)
    second = export_clip(source, export_card(source), directory, force=True)
    assert not second["cached"]
    assert second["video_path"] == first["video_path"]
    assert len(encodes) == 1


@pytest.mark.parametrize("cuts,expected_times,duration", [([[.4, .9]], [0., .1, .4, 1.], 1.1),
                                                         ([[.4, 1.6]], [0., .1], .4),
                                                         ([[0., .4]], [0., .5, 1.1], 1.2)])
def test_variable_rate_frame_timing_and_final_hold(tmp_path, cuts, expected_times, duration):
    path = tmp_path / "variable.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10:duration=2",
                    "-vf", "select=eq(n\\,0)+eq(n\\,1)+eq(n\\,4)+eq(n\\,9)+eq(n\\,15)",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "1", "-bf", "0", str(path)], capture_output=True, check=True)
    rec = load_recording(str(path))
    result = export_clip(rec, export_card(rec, cuts=cuts), str(tmp_path / "exports"))
    encoded = load_recording(result["video_path"])
    np.testing.assert_allclose(encoded.frame_times, expected_times, atol=.0001)
    assert encoded.meta["duration_s"] == pytest.approx(duration, abs=.0001)
    assert result["sensors_path"] is None and result["sensor_status"] == "no motion data"


def test_face_region_blur_preserves_pixels_outside_region(source, tmp_path):
    card = export_card(source)
    card["privacy_blur_intervals"] = [[0., 4.]]
    table = pd.DataFrame({"t": [.5], "face_boxes": [[[.25, .25, .3, .3]]]})
    result = export_clip(source, card, str(tmp_path / "exports"), table)
    before, after = frame_at(source.video_path, .5), frame_at(result["video_path"], .5)
    region = result["plan"]["privacy_regions"][0]
    x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
    def sharpness(frame): return cv2.Laplacian(cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert sharpness(after) < sharpness(before) / 3
    assert np.abs(after[-8:].astype(float)-before[-8:].astype(float)).mean() < 10
    assert result["eligible_for_training"]


def test_pending_copy_preserves_context_and_rejected_clip_is_omitted(source, tmp_path):
    card = export_card(source, "REVIEW", [[1., 2.]])
    card["privacy_blur_intervals"] = [[0., 1.]]
    pending = export_clip(source, card, str(tmp_path / "exports"))
    assert pending["export_status"] == "retained_pending" and not pending["eligible_for_training"]
    assert pending["output_duration_s"] == 4. and pending["plan"]["applied_cuts"] == []
    card["status"] = "REJECT"
    rejected = export_clip(source, card, str(tmp_path / "exports"))
    assert rejected["export_status"] == "not_exported" and "video_path" not in rejected
    card["status"], card["fixture_kind"] = "ACCEPT", "CONTROLLED DEMO FIXTURE"
    duplicate = export_clip(source, card, str(tmp_path / "exports"))
    collection = write_collection_manifest({"pending": pending, "reject": rejected, "copy": duplicate}, str(tmp_path / "exports"))
    assert collection["accepted_count"] == 0 and collection["pending_count"] == 1 and collection["omitted_count"] == 2

def test_pending_local_trim_is_playable_and_kept_out_of_accepted_collection(source,tmp_path):
    card=export_card(source,'REVIEW',[[1.,2.]])
    card['completion_required']=True
    card['checks'].append({'name':'process.local_unusable_time','passed':True,
                          'reason':'exact black frames','flags':[[.5,2.5,'black frames']]})
    result=export_clip(source,card,str(tmp_path/'exports'))
    assert result['export_status']=='retained_pending' and not result['eligible_for_training']
    assert result['output_duration_s']==pytest.approx(3.)
    assert result['video_encoding']['codec_name']=='h264'
    assert len(load_recording(result['video_path']).frame_times)==30
    collection=write_collection_manifest({'pending':result},str(tmp_path/'exports'))
    assert collection['pending_count']==1 and collection['accepted_count']==0


def test_rotation_is_baked_once_into_output_pixels(source, tmp_path):
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", source.video_path, "-c", "copy", str(rotated)], capture_output=True, check=True)
    rec = load_recording(str(rotated))
    assert rec.meta["rotation"] == 90
    result = export_clip(rec, export_card(rec), str(tmp_path / "exports"))
    encoded = load_recording(result["video_path"])
    assert encoded.meta["rotation"] == 0
    assert (encoded.meta["display_width"], encoded.meta["display_height"]) == (96, 160)


def test_collection_updates_report_with_corrected_player_and_sensor_links(source, tmp_path):
    sensor_path = Path(source.video_path).with_name("sample_sensors.csv")
    source.imu.to_csv(sensor_path, index=False)
    table_path = tmp_path / "frames.json"
    table_path.write_text(json.dumps([{"t": 0., "contract": {}}]))
    card = {**export_card(source), "video_path": source.video_path, "sensor_present": True,
            "table_path": str(table_path), "card_path": str(tmp_path/"card.json"), "rerun_path": str(tmp_path/"timeline.rrd"),
            "quality_score": 100., "usable_seconds": 4., "claude_status": "fixture", "thumbnails": []}
    analysis = {"map": [], "scene_distinctness": {"sample": 0.}, "duplicate_candidates": [], "value_ranking": [], "note": "synthetic fixture"}
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({"cards": {"sample": card}, "analysis": analysis}))
    result = export_collection(str(batch_path), str(tmp_path/"exports"))
    assert result["accepted_count"] == 1 and not result["errors"]
    page = (tmp_path/"index.html").read_text()
    assert "corrected_sample" in page and "Retimed sensors" in page
    assert "Original → edited timestamp map" in page and "Edit manifest" in page
    assert "<h2>Corrected collection</h2>" not in page
    saved = json.loads(batch_path.read_text())
    assert saved["cards"]["sample"]["export"]["eligible_for_training"]
