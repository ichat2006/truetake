import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from truetake import batch
from truetake.models import CheckResult
from test_verify_gyro import fixture_table, MAPPING


PROFILE = {"schema_version": 1, "status": "calibrated", "axis_mapping": MAPPING,
           "min_match_score": 0.9, "train_clip_ids": ["synthetic_train"]}


def test_batch_runs_calibrated_verification_even_on_quality_rejection(tmp_path, monkeypatch):
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "bad.mp4").write_bytes(b"fixture")
    (folder / "bad_sensors.csv").write_bytes(b"fixture")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(PROFILE))
    table = fixture_table()
    table["contract"] = pd.Series([None] * len(table), dtype=object)
    rec = SimpleNamespace(imu=pd.DataFrame(), meta={"rotation": 0})
    monkeypatch.setattr(batch, "load_contract", lambda *args: {"schema_version": 2, "task": "fold"})
    monkeypatch.setattr(batch, "load_recording", lambda *args: rec)
    monkeypatch.setattr(batch, "build_table", lambda *args: table.copy())
    monkeypatch.setattr(batch, "fingerprint", lambda *args: "test")
    monkeypatch.setattr(batch, "load_visual_model", lambda *args: None)
    monkeypatch.setattr(batch, "detect_onepass", lambda *args: table.copy())
    monkeypatch.setattr(batch, "detect_claude", lambda table, *args, **kwargs: table)
    calls = []

    def flow(rec, table):
        calls.append("flow")
        return table

    monkeypatch.setattr(batch, "detect_flow", flow)
    monkeypatch.setattr(batch, "check_quality", lambda table: ([], {"status": "REJECT", "quality_score": 0., "usable_seconds": 0.}))
    monkeypatch.setattr(batch, "analyze_batch", lambda *args: {"scene_distinctness": {}})
    monkeypatch.setattr(batch, "save_recording", lambda *args: None)
    monkeypatch.setattr(batch, "save_batch_map", lambda *args: None)
    monkeypatch.setattr(batch, "write_batch_report", lambda *args: "report.html")
    result = batch.run_batch(str(folder), "contract.yaml", str(tmp_path / "output"),
                             detector_mode="onepass", gyro_profile=str(profile_path))
    assert calls == ["flow"]
    card = result["cards"]["bad"]
    assert card["status"] == "REJECT"
    assert card["verification"]["passed"] is True
    assert card["gyro_alignment"]["mapping_calibrated"]


def test_single_clip_selection_does_not_visit_other_videos(tmp_path, monkeypatch):
    folder = tmp_path / "input"
    folder.mkdir()
    for name in ("wanted", "unwanted"):
        (folder / (name + ".mp4")).write_bytes(b"fixture")
    monkeypatch.setattr(batch, "load_contract", lambda *args: {"schema_version": 2, "task": "fold"})
    monkeypatch.setattr(batch, "fingerprint", lambda *args: "test")
    visited = []
    def load(video, sensors):
        visited.append(video)
        raise ValueError("fixture: stop before inference")
    monkeypatch.setattr(batch, "load_recording", load)
    monkeypatch.setattr(batch, "analyze_batch", lambda *args: {"scene_distinctness": {}})
    monkeypatch.setattr(batch, "write_batch_report", lambda *args: (_ for _ in ()).throw(AssertionError("no report during run")))
    result = batch.run_batch(str(folder), "unused.yaml", str(tmp_path / "output"), clip_ids=["wanted"], render=False)
    assert len(visited) == 1 and Path(visited[0]).stem == "wanted"
    assert result["report"] is None


def test_confirmed_mismatch_blocks_training_eligibility():
    quality = {"status": "ACCEPT", "usable_intervals": [(0., 20.)], "usable_seconds": 20.}
    result = batch.apply_verification_quality(quality, [CheckResult("verification.gyro", False, 0.1, "gyro does not match video", [])])
    assert result["status"] == "REJECT"
    assert result["usable_seconds"] == 0.
    assert result["usable_intervals"] == []
    assert quality["status"] == "ACCEPT"


def test_missing_sensors_keep_exact_no_motion_reason():
    table, checks = batch.verify_recording(SimpleNamespace(imu=None, meta={"rotation": 0}), fixture_table(), PROFILE)
    assert checks[0].passed is None
    assert checks[0].reason == "no motion data"
