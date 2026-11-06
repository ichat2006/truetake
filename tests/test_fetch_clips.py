"""Offline tests: no production corpus writes and no downloads."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load_tool("fetch_clips")
ha = load_tool("ha_to_csv")


def test_write_new_never_overwrites(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_bytes(b"original")
    fetch.write_new(path, b"original")
    with pytest.raises(FileExistsError):
        fetch.write_new(path, b"replacement")
    assert path.read_bytes() == b"original"


def sensor_files(tmp_path):
    times = np.arange(999, 3002, 1, dtype=float)
    gyro = pd.DataFrame({"Milliseconds": times, "GyroX": np.sin(times / 100),
                         "GyroY": 0.2, "GyroZ": -0.1})
    # Distinct gyro/accel clocks ensure interpolation is exercised.
    accel_times = np.arange(998.5, 3003, 2)
    accel = pd.DataFrame({"Milliseconds": accel_times, "AcclX": accel_times / 1000,
                         "AcclY": 0.0, "AcclZ": 9.81})
    g, a = tmp_path / "gyro.csv", tmp_path / "accel.csv"
    gyro.to_csv(g, index=False)
    accel.to_csv(a, index=False)
    return g, a


def test_sensor_cut_origin_units_axes_interpolation(tmp_path):
    g, a = sensor_files(tmp_path)
    frame = fetch.normalize_epic(g, a, 1.0, 3.0)
    assert list(frame) == ["t", "gx", "gy", "gz", "ax", "ay", "az"]
    assert frame.t.iloc[0] == 0.0
    assert frame.t.iloc[-1] == pytest.approx(1.999)
    assert np.allclose(frame.ax, frame.t + 1)
    assert np.allclose(frame.az, 9.81)
    assert np.allclose(frame.gy, 0.2)
    assert np.allclose(frame.gz, -0.1)


@pytest.mark.parametrize("kind", ["gyro_gap", "accel_gap", "duplicate", "nan", "coverage"])
def test_invalid_sensors_fail_closed(tmp_path, kind):
    g, a = sensor_files(tmp_path)
    target = a if kind == "accel_gap" else g
    frame = pd.read_csv(target)
    if kind in {"gyro_gap", "accel_gap"}:
        frame = frame[~frame.Milliseconds.between(1500, 1800)]
    elif kind == "duplicate":
        frame.loc[10, "Milliseconds"] = frame.loc[9, "Milliseconds"]
    elif kind == "nan":
        frame.loc[10, "GyroX"] = np.nan
    else:
        frame = frame[frame.Milliseconds > 1500]
    frame.to_csv(target, index=False)
    with pytest.raises(ValueError):
        fetch.normalize_epic(g, a, 1, 3)


def test_suspicious_degrees_warn_but_never_guess_conversion(tmp_path):
    g, a = sensor_files(tmp_path)
    frame = pd.read_csv(g)
    frame["GyroY"] = 90.0
    frame.to_csv(g, index=False)
    with pytest.warns(UserWarning, match="deg/s"):
        result = fetch.normalize_epic(g, a, 1, 3)
    assert (result.gy == 90.0).all()


def test_overlapping_annotations_do_not_double_count():
    assert fetch.coverage([(0, 50), (10, 55), (59, 100)], 0) == 56


def test_pinned_plan_is_disjoint_and_manifest_agrees():
    plan = json.loads(fetch.PLAN.read_text())
    rows = pd.read_csv(fetch.MANIFEST, keep_default_na=False)
    epic = [job for job in plan["jobs"] if job["source"] == "epic"]
    assert len(epic) == 16
    assert len({job["participant"] for job in epic}) == 8
    assert sum(job["task"] == "washing" for job in epic) == 12
    assert sum(job["task"] == "cooking_no_wash" for job in epic) == 4
    for person in {job["participant"] for job in epic}:
        subset = [job for job in epic if job["participant"] == person]
        assert len(subset) == 2
    # Hub/own now use the explicitly requested clip-level split; actors unknown.
    for _, group in rows[rows.source == "epic"].groupby("participant"):
        assert group.split.nunique() == 1
    for video in {job["video_id"] for job in epic}:
        assert len({job["split"] for job in epic if job["video_id"] == video}) == 1
    assert set(rows.goal_met_expected) == {"unknown"}
    assert set(rows.label_status) == {"unverified"}
    assert list(rows) == fetch.FIELDS
    assert fetch.manifest_bytes(plan["jobs"]) == fetch.MANIFEST.read_bytes()
    assert sum(job["source"] == "hub" for job in plan["jobs"]) == 7
    assert sum(job["source"] == "own" for job in plan["jobs"]) == 6
    assert set(rows[(rows.source == "own") & (rows.split == "test")].clip_id) == fetch.OWN_TEST_CLIPS
    assert set(rows[(rows.source == "hub") & (rows.split == "test")].clip_id) == {
        "hub_41_iPhone_Fold_Clothes_Take1", "hub_42_iPhone_Fold_Clothes_Take2", "hub_09_iPhone_Wipe_The_Countertop"}
    assert sum(rows.split == "train") == 18
    assert sum(rows.split == "test") == 11
    for _, group in rows.groupby("video_path"):
        assert group.split.nunique() == 1
    alias = rows.set_index("clip_id").loc["basic_fold_0_controlled_reupload"]
    original = rows.set_index("clip_id").loc["basic_fold_0"]
    assert alias.video_path == original.video_path
    assert alias.split == original.split == "train"
    assert alias.attack == "controlled_reupload"


def test_requested_split_migration_is_idempotent_and_preserves_media(tmp_path, monkeypatch):
    original_plan = json.loads(fetch.PLAN.read_text())
    manifest = tmp_path / "data/manifest.csv"
    plan_path = tmp_path / "tools/clip_sources.json"
    fetch.write_new(manifest, fetch.MANIFEST.read_bytes())
    fetch.write_new(plan_path, fetch.PLAN.read_bytes())
    own_jobs = [job for job in original_plan["jobs"] if job["source"] == "own"]
    protected = tmp_path / "data/raw/existing.mp4"
    fetch.write_new(protected, b"original media")
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(fetch, "PLAN", plan_path)
    monkeypatch.setattr(fetch, "MANIFEST", manifest)
    monkeypatch.setattr(fetch, "existing_jobs", lambda: own_jobs)
    before = manifest.read_bytes(), plan_path.read_bytes()
    fetch.apply_requested_split()
    fetch.apply_requested_split()
    assert before == (manifest.read_bytes(), plan_path.read_bytes())
    assert protected.read_bytes() == b"original media"


def test_split_update_refuses_reviewed_labels(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    manifest = tmp_path / "manifest.csv"
    plan_path.write_bytes(fetch.PLAN.read_bytes())
    rows = pd.read_csv(fetch.MANIFEST, keep_default_na=False)
    rows.loc[0, "label_status"] = "verified"
    rows.to_csv(manifest, index=False)
    before = manifest.read_bytes()
    monkeypatch.setattr(fetch, "PLAN", plan_path)
    monkeypatch.setattr(fetch, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="reviewed labels"):
        fetch.apply_requested_split()
    assert manifest.read_bytes() == before


def test_missing_accel_is_not_eligible(monkeypatch):
    monkeypatch.setattr(fetch, "PARTICIPANTS", ("P01",))
    rows = pd.DataFrame([dict(participant_id="P01", video_id="P01_100", start_timestamp="00:00:00",
                              stop_timestamp="00:03:00", verb="wash", narration="wash plate", narration_id="n1")])
    with pytest.raises(ValueError, match="Cannot select"):
        fetch.select_epic(rows, {"P01/videos/P01_100.MP4", "P01/meta_data/P01_100-gyro.csv"})


def test_existing_partial_output_never_downloaded_or_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"do not change")
    with pytest.raises(FileExistsError):
        fetch.cached({"video_path": "clip.mp4", "sensors_path": ""})
    assert path.read_bytes() == b"do not change"


def test_ha_inspection_prints_columns_and_stops_without_output(tmp_path, capsys):
    path = tmp_path / "head.parquet"
    pq.write_table(pa.table({"unknown_timestamp": [1], "head_field": [2]}), path)
    before = path.read_bytes()
    assert ha.inspect_columns(path) == ["unknown_timestamp", "head_field"]
    assert "STOP" in capsys.readouterr().out
    assert path.read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1


def test_hub_license_gate_happens_before_any_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("download must not run before license acceptance")
    monkeypatch.setattr(fetch, "hf_download", forbidden)
    with pytest.raises(SystemExit) as result:
        fetch.main(["fetch", "--source", "hub", "--limit", "1"])
    assert result.value.code == 2


def test_stationary_windows_skip_before_video_download(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    times = np.arange(0, 60001, 5)
    gyro = tmp_path / "gyro.csv"
    accel = tmp_path / "accel.csv"
    pd.DataFrame({"Milliseconds": times, "GyroX": 0, "GyroY": 0, "GyroZ": 0}).to_csv(gyro, index=False)
    pd.DataFrame({"Milliseconds": times, "AcclX": 0, "AcclY": 0, "AcclZ": 9.81}).to_csv(accel, index=False)
    downloads = []

    def fake_download(repo, revision, filename, directory):
        downloads.append(filename)
        assert not filename.endswith(".MP4"), "stationary clip should not download video"
        return gyro if filename.endswith("-gyro.csv") else accel

    monkeypatch.setattr(fetch, "hf_download", fake_download)
    job = dict(clip_id="P01_100_000_060", video_id="P01_100", participant="P01", start=0, end=60,
               video_path="data/clips/example.mp4", sensors_path="data/clips/example_sensors.csv")
    assert fetch.fetch_epic_group([job]) == [job["clip_id"]]
    assert len(downloads) == 2
    assert not (tmp_path / job["video_path"]).exists()
    assert gyro.exists() and accel.exists()
