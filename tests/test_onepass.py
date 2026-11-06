from contextlib import contextmanager

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from truetake.detect_onepass import detect_onepass, scheduled_rows
from truetake.frame_table import build_table
from truetake.models import Recording


def setup_models(monkeypatch, frames):
    calls = {"decodes": 0, "batches": [], "hands": [], "closed": False}
    def iterator(rec, table):
        calls["decodes"] += 1
        for row in table.index:
            yield row, frames[row].copy()
    @contextmanager
    def people_models(path):
        try:
            yield None, None
        finally:
            calls["closed"] = True
    def people(frame, time, hands, faces, previous, previous_time, run_hands=True):
        calls["hands"].append(run_hands)
        result = {"people_status": "ok" if run_hands else "skipped_dark", "face_status": "ok",
                  "face_confidence": .97, "face_boxes": [[.1, .1, .2, .2]]}
        if run_hands:
            result.update(hand_landmarks=[], hand_count=0, hands_inside=0.)
        return result
    class Model:
        def encode_image(self, batch):
            calls["batches"].append(len(batch))
            return batch.clone()
    def preprocess(image):
        return torch.tensor([np.asarray(image).mean() / 255., 1.], dtype=torch.float32)
    monkeypatch.setattr("truetake.detect_onepass.iter_sampled_frames", iterator)
    monkeypatch.setattr("truetake.detect_onepass.people_models", people_models)
    monkeypatch.setattr("truetake.detect_onepass.observe_people", people)
    rec = Recording("synthetic", "unused.mp4", np.arange(len(frames)) / 10,
                    {"duration_s": len(frames) / 10, "rotation": 0}, None)
    visual = (Model(), preprocess, torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))
    return rec, visual, calls


def test_one_decode_batches_and_original_evidence_without_hands(tmp_path, monkeypatch):
    rng = np.random.default_rng(123)
    frames = [rng.integers(40, 255, (18, 32, 3), dtype=np.uint8) for _ in range(200)]
    rec, visual, calls = setup_models(monkeypatch, frames)
    source = build_table(rec)
    result = detect_onepass(rec, source, visual, str(tmp_path), "unused")
    stats = result.iloc[0].detector_stats
    assert calls["decodes"] == stats["analysis_decode_passes"] == 1
    assert stats["evidence_decode_passes"] == 0
    assert stats["face_frames"] == stats["hand_frames"] == 100
    assert stats["clip_frames"] == 20 and stats["multipass_clip_frames_at_2hz"] == 40
    assert calls["batches"] == [8, 8, 4]
    assert calls["closed"]
    assert "brightness" not in source
    assert not list(tmp_path.glob(".evidence-*"))
    for row, sample in result[result.claude_selected].iterrows():
        ok, expected = cv2.imencode(".jpg", frames[row], [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        from pathlib import Path
        assert Path(sample.thumbnail_path).read_bytes() == expected.tobytes()
    expected_brightness = [float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) for frame in frames]
    np.testing.assert_allclose(result.brightness, expected_brightness)


def test_black_gates_preserve_privacy_and_unknowns(tmp_path, monkeypatch):
    frames = [np.zeros((18, 32, 3), np.uint8)] * 10 + [np.full((18, 32, 3), 100, np.uint8)] * 10
    rec, visual, calls = setup_models(monkeypatch, frames)
    result = detect_onepass(rec, build_table(rec), visual, str(tmp_path), "unused")
    stats = result.iloc[0].detector_stats
    assert stats["face_frames"] == 10
    assert stats["hands_skipped_dark"] == 5 and stats["clip_skipped_dark"] == 1
    assert result.face_confidence.dropna().eq(.97).all()
    assert pd.isna(result.at[0, "hand_count"]) and result.at[0, "hand_landmarks"] is None
    assert result.at[0, "clip_status"] == "skipped_dark" and result.at[0, "clip_embedding"] is None
    assert result.at[0, "claude_selected"] and result.claude_selected.iloc[-1]
    assert result[result.clip_status == "ok"].shape[0] == 1


def test_failure_closes_models_and_removes_temporary_images(tmp_path, monkeypatch):
    rec, visual, calls = setup_models(monkeypatch, [np.full((18, 32, 3), 100, np.uint8)] * 10)
    def fail(*args, **kwargs):
        raise ValueError("simulated inference failure")
    monkeypatch.setattr("truetake.detect_onepass.observe_people", fail)
    with pytest.raises(ValueError, match="simulated"):
        detect_onepass(rec, build_table(rec), visual, str(tmp_path), "unused")
    assert calls["closed"] and not list(tmp_path.glob(".evidence-*"))


def test_variable_rate_scheduling_does_not_invent_frames():
    table = pd.DataFrame({"t": [0., .3, .7], "sample_end": [.3, .7, 1.]}, index=[4, 8, 12])
    assert scheduled_rows(table, 5) == {4, 8, 12}
    assert scheduled_rows(table, 1) == {4}
