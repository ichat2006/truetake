import json
from pathlib import Path

from truetake.report import write_batch_report
from truetake.compare import compare_reviews

def test_test_agreement_does_not_invent_unknown_labels_or_score_reviews():
    import importlib.util
    spec=importlib.util.spec_from_file_location('agreement',Path('docs/final_agreement.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'clip_id':name,'split':split,'source':'own','label_status':'unverified',
           'goal_met_expected':goal,'attack':attack}
          for name,split,goal,attack in [('yes','test','yes','none'),('unknown','test','unknown','none'),
              ('review','test','no','none'),('swap','test','yes','swap'),('train','train','yes','none')]]
    cards={name:{'verdict':verdict} for name,verdict in [('yes','APPROVED'),('unknown','APPROVED'),
            ('review','NEEDS REVIEW'),('swap','REJECTED'),('train','REJECTED')]}
    result=module.compare(rows,cards)
    assert len(result['rows'])==4 and '2 of 2' in result['line']
    assert 'separately (1)' in result['line']
    assert next(r for r in result['rows'] if r['clip_id']=='unknown')['agree'] is None

def test_lightweight_bundle_removes_other_videos_and_large_artifact_links(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('packaging',Path('docs/package_submission.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'selected.mp4').write_bytes(b'fixture')
    text="<article id='selected'><video id='video_selected' src='selected.mp4'></video></article><article id='other'><video id='video_other' src='../raw.mp4'></video></article><p><a href='../huge.rrd'>Rerun recording</a> · <a href='../frames.json'>Per-frame table</a></p>"
    text=module.finalize_html(text,{'selected'})
    assert 'video_other' not in text and 'huge.rrd' not in text
    assert module.LinkAudit(tmp_path).validate(text)==['selected.mp4']
    import pytest
    with pytest.raises(ValueError,match='escaping'):
        module.LinkAudit(tmp_path).validate("<img src='../outside.jpg'>")


def test_report_is_offline_escapes_text_and_links_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("truetake.report.browser_preview", lambda video, destination, spans: video)
    card = {"status": "REVIEW", "license":"own", "checks": [{"name": "goal.folded", "passed": None, "reason": "<unknown>", "flags": []}],
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
    assert "href='clip.rrd'" not in text  # No broken link to an ungenerated recording.
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
