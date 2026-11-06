import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from truetake.detect_claude import detect_claude
from truetake.evidence import select_evidence, candidate_windows, route_followups
from test_lab_quality import lab_table


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 100, 600])
def test_bounded_evidence_keeps_start_end_and_does_not_require_hands(rows):
    table = pd.DataFrame({"t": np.arange(rows) / 10, "sample_end": np.arange(1, rows + 1) / 10,
                          "hand_count": 0, "brightness": 100., "blur_var": 100., "frame_change": .1})
    result = select_evidence(table)
    selected = result[result.claude_selected]
    assert len(selected) == min(8, rows)
    assert selected.index[0] == 0 and selected.index[-1] == rows - 1
    assert "claude_selected" not in table
    if rows >= 100:
        duration = table.sample_end.iloc[-1]
        for quarter in range(4):
            assert ((selected.t >= quarter * duration / 4) & (selected.t < (quarter + 1) * duration / 4)).any()


def test_dark_windows_are_hints_and_missing_samples_break_runs():
    table = pd.DataFrame({"t": np.arange(50) / 10, "sample_end": np.arange(1, 51) / 10, "brightness": 100.})
    table.loc[10:29, "brightness"] = 0.
    windows = candidate_windows(table)
    assert len(windows) == 1
    assert windows[0]["start_s"] == 1. and windows[0]["end_s"] == 3.
    table.loc[18:22, "brightness"] = np.nan
    assert candidate_windows(table) == []


def test_optional_followups_are_deferred_but_required_uncertainty_gets_evidence():
    table = lab_table()
    review = table.iloc[0].claude_review
    review["follow_up_requests"] = [{"start_s": 1., "end_s": 4., "question": "Check optional hands"}]
    routed = route_followups(review, table.iloc[0].contract)
    assert routed["follow_up_requests"] == []
    assert routed["deferred_follow_up_requests"] == review["follow_up_requests"]
    review["goal"]["verdict"] = "unsure"
    assert route_followups(review, table.iloc[0].contract)["follow_up_requests"]
    review["goal"]["verdict"] = "no"
    assert route_followups(review, table.iloc[0].contract)["follow_up_requests"] == []


def test_focused_call_sends_only_selected_images_and_counts_real_usage(tmp_path, monkeypatch):
    table = lab_table()
    response_review = deepcopy(table.iloc[0].claude_review)
    table["claude_review"] = None
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"fake image for mocked SDK")
    table["thumbnail_path"] = str(image)
    table = select_evidence(table)
    prompt = tmp_path / "focused.md"
    prompt.write_text("Contract once: {contract_yaml}\nHints: {detector_evidence}")
    captured = []
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        @property
        def messages(self): return self
        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=123, output_tokens=456),
                                   content=[SimpleNamespace(type="text", text=json.dumps(response_review))])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "UNIT_TEST_NOT_A_REAL_KEY")
    monkeypatch.setattr("truetake.detect_claude.anthropic.Anthropic", lambda **kwargs: Client())
    first = detect_claude(table, str(tmp_path / "cache"), str(prompt))
    assert first.iloc[0].claude_status == "ok"
    images = [part for part in captured[0]["messages"][0]["content"] if part["type"] == "image"]
    assert len(images) == 8
    usage = first.iloc[0].claude_usage
    assert usage["actual_api_calls"] == 1 and usage["actual_images_sent"] == 8
    assert usage["actual_input_tokens"] == 123 and usage["actual_output_tokens"] == 456
    cached = detect_claude(table, str(tmp_path / "cache"), str(prompt))
    assert cached.iloc[0].claude_status == "cached"
    assert cached.iloc[0].claude_usage["actual_api_calls"] == 0
    assert len(captured) == 1
    changed = table.copy()
    changed.iloc[0].evidence_plan["selector_version"] += 1
    assert detect_claude(changed, str(tmp_path / "cache"), str(prompt)).iloc[0].claude_status == "ok"
    assert len(captured) == 2
