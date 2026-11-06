from truetake.compare import compare_reviews


def card(status="ACCEPT", inputs=100, outputs=50, images=12):
    return {"status": status, "claude_review": {"provenance": {"input_tokens": inputs, "output_tokens": outputs,
            "api_calls": 1, "frame_ids": list(range(images)), "images_sent": images}}}


def test_compares_paid_usage_not_free_cache_replay_and_excludes_controlled_copy():
    baseline = {"cards": {"a": card(), "copy": {**card(), "fixture_kind": "CONTROLLED DEMO FIXTURE"}}}
    focused = {"a": card(inputs=80, outputs=40, images=8)}
    focused["a"]["claude_usage"] = {"actual_input_tokens": 0, "actual_api_calls": 0}
    result = compare_reviews(baseline, focused)
    assert result["real_clips"] == result["status_matches"] == 1
    assert result["input_token_reduction_percent"] == result["output_token_reduction_percent"] == 20
    assert result["focused"]["calls"] == 1


def test_missing_or_inconclusive_reviews_do_not_manufacture_savings():
    baseline = {"cards": {"a": card(), "b": card()}}
    result = compare_reviews(baseline, {"a": {"status": "REVIEW", "claude_review": None}})
    assert result["missing_clips"] == ["b"]
    assert result["incomplete_usage_clips"] == ["a"]
    assert result["status_matches"] == 0
    assert result["input_token_reduction_percent"] is None
