import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from truetake.models import Recording
from test_verify_gyro import fixture_table


spec = importlib.util.spec_from_file_location("gyro_swap_eval", Path(__file__).resolve().parents[1] / "tools/gyro_swap_eval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_shifted_sine_passes_while_unrelated_swap_scores_low(tmp_path):
    real = fixture_table()
    swapped = real.copy()
    swapped[["gyro_gx", "gyro_gy", "gyro_gz"]] = np.random.default_rng(98).normal(size=(len(real), 3))
    rows = evaluation.score_pair(real, swapped)
    assert rows[0]["score"] > 0.95
    assert abs(rows[0]["offset_s"] - 0.7) < 0.100001
    assert rows[1]["score"] < 0.35
    suggestion = evaluation.suggest_threshold(rows)
    assert suggestion["usable_separation"]
    assert rows[1]["score"] < suggestion["suggested_threshold"] < rows[0]["score"]
    path = tmp_path / "hist.png"
    evaluation.plot_scores(rows, path)
    assert path.read_bytes().startswith(b"\x89PNG")


def test_overlapping_groups_are_not_promoted_to_calibration():
    rows = [{"pairing": pairing, "status": "measured", "score": value}
            for pairing, values in [("real", [0.4, 0.8]), ("swapped", [0.3, 0.7])] for value in values]
    assert not evaluation.suggest_threshold(rows)["usable_separation"]


def test_evaluation_never_opens_test_media_and_swaps_other_participants(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    entries = []
    for clip, person, split in [("a", "P01", "train"), ("b", "P07", "train"), ("forbidden", "P30", "test")]:
        entries.append(dict(clip_id=clip, participant=person, split=split,
                            video_path=clip + ".mp4", sensors_path=clip + ".csv", attack="none"))
        if split == "train":
            (tmp_path / (clip + ".mp4")).write_bytes(b"fixture")
            (tmp_path / (clip + ".csv")).write_bytes(b"fixture")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    opened = []

    def fake_load(video, sensors):
        assert "forbidden" not in video and "forbidden" not in sensors
        opened.append(video)
        table = fixture_table()
        imu = pd.DataFrame({"t": table.t, "gx": table.gyro_gx, "gy": table.gyro_gy, "gz": table.gyro_gz,
                            "ax": 0., "ay": 0., "az": 9.81})
        return Recording(Path(video).stem, video, table.t.to_numpy(), {"duration_s": 20., "rotation": 0}, imu)

    monkeypatch.setattr(evaluation, "load_recording", fake_load)
    monkeypatch.setattr(evaluation, "flow_table", lambda *args: fixture_table())
    report = evaluation.run_evaluation(manifest, tmp_path / "outputs/scores.csv")
    assert len(opened) == 2
    assert report["train_clip_ids"] == ["a", "b"]
    assert all(pair["participant"] != pair["other_participant"] for pair in report["pairings"])
    assert len(pd.read_csv(tmp_path / "outputs/scores.csv")) == 4
