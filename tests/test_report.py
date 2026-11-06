import json
from pathlib import Path

from truetake.report import write_batch_report
from truetake.compare import compare_reviews


def test_report_is_offline_escapes_text_and_links_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("truetake.report.browser_preview", lambda video, destination, spans: video)
    card = {"status": "REVIEW", "checks": [{"name": "goal.folded", "passed": None, "reason": "<unknown>", "flags": []}],
            "thumbnails": [], "quality_score": 70., "usable_seconds": 0., "claude_status": "missing_api_key",
            "rerun_path": str(tmp_path / "clip.rrd"), "card_path": str(tmp_path / "clip.json"),
            "table_path": str(tmp_path / "frames.json"), "video_path": str(tmp_path / "video.mp4")}
    analysis = {"map": [{"clip_id": "clip", "status": "REVIEW", "x": 0., "y": 0.}],
                "scene_distinctness": {"clip": 50.}, "duplicate_candidates": [], "value_ranking": [], "note": "<pending>"}
    analysis["evidence_comparison"] = compare_reviews({"cards": {"clip": card}}, {"clip": card})
    analysis["detector_mode"] = "onepass"
    analysis["detector_totals"] = {"sampled_frames": 10, "hand_frames": 5, "face_frames": 5, "clip_frames": 1}
    path = write_batch_report({"clip": card}, analysis, str(tmp_path))
    text = Path(path).read_text()
    assert "&lt;unknown&gt;" in text
    assert "<unknown>" not in text
    assert "href='clip.rrd'" in text
    assert "No approved clips" in text
    assert "Claude evidence comparison" not in text
    assert "Local detector funnel" not in text
    assert "Corrected collection" not in text
    assert "Local detectors:" not in text and "input /" not in text
    assert "Each point is one clip" in text
    assert "axes are not quality scores" in text
    assert "<script src=" not in text
    assert "id='video_clip'" in text
    saved = json.loads((tmp_path / "batch.json").read_text())
    assert saved["cards"]["clip"]["status"] == "REVIEW"
    assert saved["analysis"]["evidence_comparison"] == analysis["evidence_comparison"]
