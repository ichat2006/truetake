import json
from pathlib import Path

from truetake.presentation import badge, render_report
from truetake.value import suggest_next


def card(root, name="clip", status="ACCEPT"):
    return {"clip_id": name, "status": status, "quality_score": 100., "usable_seconds": 12., "duration_s": 20.,
            "sensor_present": False, "checks": [], "thumbnails": [{"t": 1., "path": str(root / "thumb.jpg")}],
            "video_path": str(root / "video.mp4"), "preview_path": str(root / "video.mp4"),
            "rerun_path": str(root / "clip.rrd"), "card_path": str(root / "card.json"),
            "table_path": str(root / "frames.json"), "export": {}, "verification": {"passed": None}}


def test_badges_describe_saved_status_and_real_applied_cuts(tmp_path):
    original = card(tmp_path)
    assert badge(original) == "APPROVED"
    assert badge({**original, "status": "REJECT"}) == "REJECTED"
    assert badge({**original, "status": "REVIEW"}) == "NOT VERIFIED"
    assert badge({**original, "fixture_kind": "controlled duplicate"}) == "DUPLICATE"
    assert badge({**original, "excluded_intervals": [[10., 18.]]}) == "APPROVED"
    assert badge({**original, "export": {"plan": {"applied_cuts": [{"start_s": 10., "end_s": 18.}]}}}) == "CUT"


def test_report_has_thumbnail_map_side_by_side_players_and_actual_cut_bar(tmp_path):
    sample = card(tmp_path)
    sample["export"] = {"video_path": str(tmp_path / "trimmed.mp4"), "plan": {
        "applied_cuts": [{"start_s": 10., "end_s": 18.}], "kept_segments": []}}
    analysis = {"map": [{"clip_id": "clip", "x": 0., "y": 0.}], "film_next": "A towel on a bed", "scene_distinctness": {}}
    before = json.dumps(sample, sort_keys=True)
    text = render_report({"clip": sample}, analysis, tmp_path)
    assert "1 clips in → 1 approved" in text
    assert "Film next: A towel on a bed" in text
    assert "<image href='thumb.jpg'" in text
    assert "viewBox='0 0 1200 520'" in text
    assert "class='comparison'" in text
    assert "id='video_clip'" in text and "id='corrected_clip'" in text
    assert "left:50.000%;width:40.000%" in text
    assert "Not verified: no motion sensor data" in text
    assert "<details><summary>Technical details, scores and evidence" in text
    assert json.dumps(sample, sort_keys=True) == before


def test_plain_failure_reasons_are_escaped_and_clickable(tmp_path):
    sample = card(tmp_path, status="REJECT")
    sample["decision_reasons"] = ["goal.folded: <unsafe> garment remains unfolded."]
    sample["checks"] = [{"name": "goal.folded", "passed": False, "reason": "<unsafe>", "flags": [(3., 5., "failure")]}]
    text = render_report({"clip": sample}, {"map": []}, tmp_path)
    assert "&lt;unsafe&gt;" in text and "<unsafe>" not in text
    assert "data-seek='3.000000'" in text


def test_value_suggestion_uses_approved_tags_and_ignores_duplicates(tmp_path):
    sample = card(tmp_path)
    sample["claude_review"] = {"tags": {"garment": "tshirt"}}
    wanted = [{"garment": "tshirt"}, {"garment": "towel"}]
    suggestion = suggest_next({"clip": sample}, wanted)
    assert "towel" in suggestion and "0 approved" in suggestion


def test_public_publication_excludes_hub_from_html_json_and_assets(tmp_path):
    import importlib.util
    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "tools/build_demo_report.py"
    spec = importlib.util.spec_from_file_location("build_demo_report", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    entries = {}
    for name, source in [("own_example", "own"), ("private_hub_example", "hub")]:
        sample = card(inputs, name)
        sample["source"] = source
        sample["export"] = {"export_status": "not_exported", "eligible_for_training": False}
        sample["thumbnails"] = []
        sample["table_path"] = str(inputs / (name + ".json"))
        pd.DataFrame({"clip_embedding": [[1., 0.]]}).to_json(sample["table_path"], orient="records")
        Path(sample["preview_path"]).write_bytes(b"synthetic video fixture")
        Path(sample["rerun_path"]).write_bytes(b"synthetic rerun fixture")
        entries[name] = sample
    saved = {"cards": entries, "analysis": {"skipped": [{"clip_id": "private_hub_missing", "source": "hub"}]}}
    public = tmp_path / "public"
    builder.publish_subset(saved, public, True)
    for name in ("index.html", "batch.json"):
        assert "private_hub" not in (public / name).read_text()
    assert all("private_hub" not in str(path) for path in public.rglob("*"))
    internal = tmp_path / "internal"
    builder.publish_subset(saved, internal, False)
    assert "private_hub_example" in json.loads((internal / "batch.json").read_text())["cards"]
