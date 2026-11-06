import importlib.util
import json
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "tools/run_manifest_demo.py"
    spec = importlib.util.spec_from_file_location("manifest_resume_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "WORK", tmp_path / "outputs/work")
    monkeypatch.setattr(module, "RESULTS", tmp_path / "outputs/manifest_run")


def test_resume_skips_existing_result_before_loading_video(tmp_path, monkeypatch):
    module = runner()
    configure(module, tmp_path, monkeypatch)
    row = {"clip_id": "done", "video_path": "does-not-exist.mp4", "source": "own"}
    module.save_result(row, {"status": "ACCEPT"}, [])
    monkeypatch.setattr(module, "read_entries", lambda: [row])
    monkeypatch.setattr(module, "process_one", lambda row: pytest.fail("completed clip must not run"))
    result = module.run()
    assert result["skipped_results"] == ["done"]
    assert result["processed"] == []


def test_result_saved_before_next_clip_and_rerun_is_free(tmp_path, monkeypatch):
    module = runner()
    configure(module, tmp_path, monkeypatch)
    rows = [{"clip_id": name, "video_path": name + ".mp4", "source": "own"} for name in ("first", "second")]
    for row in rows:
        (tmp_path / row["video_path"]).write_bytes(b"fixture")
    monkeypatch.setattr(module, "read_entries", lambda: rows)
    calls = []

    def process(row):
        if row["clip_id"] == "second":
            assert (module.RESULTS / "first.json").is_file()
        calls.append(row["clip_id"])
        return module.save_result(row, {"status": "REJECT"}, [])

    monkeypatch.setattr(module, "process_one", process)
    assert module.run()["processed"] == ["first", "second"]
    assert not list(module.RESULTS.glob("*.tmp"))
    module.run()
    assert calls == ["first", "second"]


def test_failure_is_recorded_and_does_not_erase_previous_results(tmp_path, monkeypatch):
    module = runner()
    configure(module, tmp_path, monkeypatch)
    row = {"clip_id": "bad", "video_path": "bad.mp4", "source": "own"}
    (tmp_path / "bad.mp4").write_bytes(b"fixture")
    monkeypatch.setattr(module, "read_entries", lambda: [row])
    def fail(row):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(module, "process_one", fail)
    status = module.run()
    assert status["failed"] == ["bad"]
    saved = json.loads((module.RESULTS / "bad.json").read_text())
    assert saved["status"] == "failed" and saved["card"] is None
    assert "fixture failure" in saved["errors"][0]["error"]


def test_runner_selects_one_clip_and_does_not_build_group_reports(tmp_path, monkeypatch):
    module = runner()
    configure(module, tmp_path, monkeypatch)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fixture")
    row = {"clip_id": "clip", "source": "own", "split": "train", "video_path": "clip.mp4",
           "sensors_path": "", "task_expected": "folding", "attack": "none"}
    monkeypatch.setattr(module, "seed_caches", lambda *args: None)
    calls = []
    def batch(*args, **kwargs):
        calls.append(kwargs)
        card = {"clip_id": "clip", "card_path": str(tmp_path / "card.json")}
        return {"cards": {"clip": card}, "analysis": {"errors": []}}
    monkeypatch.setattr(module, "run_batch", batch)
    module.process_one(row)
    assert calls[0]["clip_ids"] == ["clip"]
    assert calls[0]["render"] is False
    assert calls[0]["export"] is True
    assert (module.RESULTS / "clip.json").is_file()
