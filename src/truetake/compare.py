"""Compare review modes on the same recordings; this is not accuracy scoring."""


def compare_reviews(baseline: dict, cards: dict[str, dict]) -> dict:
    previous = baseline.get("cards")
    if not isinstance(previous, dict):
        raise ValueError("Baseline report must contain clip cards.")
    ids = [name for name, card in previous.items() if not card.get("fixture_kind")]
    paired = [name for name in ids if name in cards and not cards[name].get("fixture_kind")]

    def metrics(collection):
        result = dict(input_tokens=0, output_tokens=0, calls=0, initial_images=0, total_images=0)
        for name in paired:
            provenance = (collection[name].get("claude_review") or {}).get("provenance", {})
            for field in ("input_tokens", "output_tokens"):
                result[field] += provenance.get(field, 0)
            result["calls"] += provenance.get("api_calls", 0)
            result["initial_images"] += len(provenance.get("initial_frame_ids", provenance.get("frame_ids", [])))
            result["total_images"] += provenance.get("images_sent", len(provenance.get("frame_ids", [])))
        return result

    a, b = metrics(previous), metrics(cards)
    clips = [{"clip_id": name, "baseline": previous[name]["status"], "focused": cards[name]["status"]} for name in paired]
    incomplete = [name for name in paired if not all((collection[name].get("claude_review") or {}).get("provenance")
                                                    for collection in (previous, cards))]
    return {"scope": "Claude evidence selection; local detector passes unchanged", "real_clips": len(paired),
            "status_matches": sum(item["baseline"] == item["focused"] for item in clips), "clips": clips,
            "missing_clips": [name for name in ids if name not in cards], "incomplete_usage_clips": incomplete,
            "baseline": a, "focused": b,
            "input_token_reduction_percent": round(100 * (1 - b["input_tokens"] / a["input_tokens"]), 1) if a["input_tokens"] and not incomplete else None,
            "output_token_reduction_percent": round(100 * (1 - b["output_tokens"] / a["output_tokens"]), 1) if a["output_tokens"] and not incomplete else None,
            "note": "Development agreement, not held-out accuracy. Counts include original paid reviews and follow-ups, not just cache replay. Controlled fixtures excluded."}
