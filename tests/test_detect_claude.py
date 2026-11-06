import pytest
from copy import deepcopy
from types import SimpleNamespace
import pandas as pd

from truetake.detect_claude import validate_review, detect_claude, parse_review, bound_followups
from test_lab_quality import lab_table


def review():
    return {"evidence": [{"id": "e1", "frame_id": 5, "t": 0.5, "observation": "fold visible"}],
            "goal": {"verdict": "yes", "reason": "visible fold", "evidence_ids": ["e1"]},
            "steps": [{"name": "smooth", "evidence_ids": ["e1"]}]}


def test_valid_grounded_response():
    assert validate_review(review(), {5: 0.5})["goal"]["verdict"] == "yes"


def test_hallucinated_frame_and_ungrounded_verdict_rejected():
    with pytest.raises(ValueError):
        validate_review(review(), {3: 0.5})
    result = review()
    result["goal"]["evidence_ids"] = []
    with pytest.raises(ValueError):
        validate_review(result, {5: 0.5})


def test_invalid_step_and_verdict_rejected():
    result = review()
    result["steps"][0]["name"] = "invented"
    with pytest.raises(ValueError):
        validate_review(result, {5: 0.5})
    result = review()
    result["goal"]["verdict"] = "maybe"
    with pytest.raises(ValueError):
        validate_review(result, {5: 0.5})


def test_json_wrappers_are_allowed_but_invalid_json_is_not_repaired():
    assert parse_review('Here is the review:\n```json\n{"evidence": []}\n```') == {"evidence": []}
    with pytest.raises(ValueError):
        parse_review('{"evidence": [}')
    assert parse_review('{"evidence": [{"frame_id": "42"}]}')["evidence"][0]["frame_id"] == 42
    response = {"follow_up_requests": [{"start_s": 10., "end_s": 25.}]}
    assert bound_followups(response)["follow_up_requests"] == [{"start_s": 17., "end_s": 25.}]


def test_targeted_review_is_cached_without_credentials(tmp_path, monkeypatch):
    table = lab_table()
    verdict = deepcopy(table.iloc[0].claude_review)
    table["claude_review"] = None
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"unit test image")
    table["thumbnail_path"] = pd.Series([None] * len(table), dtype=object)
    table.at[0, "thumbnail_path"] = str(image)
    table.at[len(table)-1, "thumbnail_path"] = str(image)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"unit test video fingerprint")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Unit test review")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "UNIT_TEST_SECRET_NOT_REAL")
    class Client:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr("truetake.detect_claude.anthropic.Anthropic", lambda **kwargs: Client())
    stages = []
    def request(client, prompt, contract, table, frames, model, stage, version):
        stages.append(stage)
        result = deepcopy(verdict)
        result["follow_up_requests"] = [{"start_s": 3., "end_s": 5., "question": "Check folding"}] if stage == "overview" else []
        return result, 100, 50
    def followup(rec, table, requests, destination):
        table.at[40, "thumbnail_path"] = str(image)
        return {40: 4.}
    monkeypatch.setattr("truetake.detect_claude._request", request)
    monkeypatch.setattr("truetake.detect_claude.followup_frames", followup)
    args = (table, str(tmp_path / "cache"), str(prompt))
    first = detect_claude(*args, rec=SimpleNamespace(video_path=str(video)), evidence_dir=str(tmp_path))
    assert first.iloc[0].claude_status == "ok"
    assert first.iloc[0].claude_review["provenance"]["api_calls"] == 2
    assert stages == ["overview", "targeted_follow_up"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    cached = detect_claude(*args, rec=SimpleNamespace(video_path=str(video)), evidence_dir=str(tmp_path))
    assert cached.iloc[0].claude_status == "cached"
    assert len(stages) == 2
    assert all("UNIT_TEST_SECRET_NOT_REAL" not in path.read_text() for path in (tmp_path / "cache").glob("*.json"))
