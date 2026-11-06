"""Frame-boundary edit plans and sensor retiming; no task judgments here."""

import math

import numpy as np
import pandas as pd

from truetake.models import Recording


TIME_TOLERANCE_S = 1e-6
FACE_PADDING = 0.03
DEFAULT_CONTEXT_S = 1.0


def merged_intervals(spans: list, duration: float) -> list[list[float]]:
    result = []
    normalized = []
    for span in spans:
        if len(span) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in span):
            raise ValueError("Edit intervals must contain two finite times.")
        a, b = map(float, span)
        if not (0 <= a < b <= duration + TIME_TOLERANCE_S and a < duration):
            raise ValueError("Edit interval outside the recording.")
        normalized.append((a, min(b, duration)))
    for a, b in sorted(normalized):
        if result and a <= result[-1][1] + TIME_TOLERANCE_S:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def privacy_regions(rec: Recording, spans: list, table: pd.DataFrame | None) -> list[dict]:
    width = rec.meta["display_width"]
    height = rec.meta["display_height"]
    regions = []
    for a, b in spans:
        boxes = []
        if table is not None and "face_boxes" in table:
            for value in table.loc[(table.t >= a) & (table.t < b), "face_boxes"]:
                for box in value if isinstance(value, list) else []:
                    if len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box) and box[2] > 0 and box[3] > 0:
                        boxes.append(box)
        if boxes:
            left = max(0., min(box[0] for box in boxes) - FACE_PADDING)
            top = max(0., min(box[1] for box in boxes) - FACE_PADDING)
            right = min(1., max(box[0] + box[2] for box in boxes) + FACE_PADDING)
            bottom = min(1., max(box[1] + box[3] for box in boxes) + FACE_PADDING)
            x, y = int(left * width) // 2 * 2, int(top * height) // 2 * 2
            w = min(math.ceil(width / 2) * 2, math.ceil(right * width / 2) * 2) - x
            h = min(math.ceil(height / 2) * 2, math.ceil(bottom * height / 2) * 2) - y
        else:
            x, y, w, h = 0, 0, width, height
        fallback = not boxes or w < 4 or h < 4
        regions.append({"start_s": a, "end_s": b, "full_frame_fallback": fallback,
                        "x": x, "y": y, "width": w, "height": h})
    return regions


def build_edit_plan(rec: Recording, card: dict, table: pd.DataFrame | None = None) -> dict:
    duration = float(rec.meta["duration_s"])
    if card["clip_id"] != rec.clip_id or abs(card["duration_s"] - duration) > TIME_TOLERANCE_S:
        raise ValueError("Review card does not match the recording identity or duration.")
    if card["status"] not in ("ACCEPT", "REVIEW", "REJECT"):
        raise ValueError("Unknown review status.")
    requested = merged_intervals(card.get("excluded_intervals", []), duration)
    regions = privacy_regions(rec, merged_intervals(card.get("privacy_blur_intervals", []), duration), table)
    fallback = any(region["full_frame_fallback"] for region in regions)
    eligible = card["status"] == "ACCEPT" and not fallback
    warnings = []
    authorized = any(check["name"] == "process.excluded_time" and check["passed"] is True for check in card.get("checks", []))
    cuts = requested if eligible and authorized else []
    if requested and not cuts:
        warnings.append("Requested cuts retained: no accepted, compiled exclusion plan for this derivative.")
    if fallback:
        warnings.append("Missing usable face boxes: full-frame blur copy is pending, not training-eligible.")
    evidence = {item["id"]: item for item in (card.get("claude_review") or {}).get("evidence", [])}
    goal_ids = (card.get("claude_review") or {}).get("goal", {}).get("evidence_ids", [])
    goal_times = [float(evidence[key]["t"]) for key in goal_ids if key in evidence]
    if any(not math.isfinite(t) or not 0 <= t < duration for t in goal_times):
        raise ValueError("Invalid completion evidence timestamp.")
    context = DEFAULT_CONTEXT_S
    if table is not None and not table.empty:
        context = float((table.iloc[0].get("contract") or {}).get("duration", {}).get("preserve_context_before_after_s", DEFAULT_CONTEXT_S))
    if not math.isfinite(context) or context < 0:
        raise ValueError("Edit context must be finite nonnegative seconds.")
    if cuts and card.get("completion_required", True) and not goal_times:
        cuts = []
        warnings.append("Completion evidence unavailable: retained requested cuts.")
    boundaries = np.append(rec.frame_times, duration)
    snapped, removed = [], np.zeros(len(rec.frame_times), dtype=bool)
    for a, b in cuts:
        first = int(np.searchsorted(boundaries, a - TIME_TOLERANCE_S, side="left"))
        end = int(np.searchsorted(boundaries, b + TIME_TOLERANCE_S, side="right") - 1)
        if first >= end:
            continue
        start_s, end_s = float(boundaries[first]), float(boundaries[end])
        if any(start_s - context <= t <= end_s + context for t in goal_times):
            warnings.append("Cut touching completion evidence retained.")
            continue
        snapped.append({"start_s": start_s, "end_s": end_s, "first_frame": first, "end_frame": end})
        removed[first:end] = True
    indices = np.flatnonzero(~removed)
    if not len(indices):
        raise ValueError("Edit plan would remove every video frame.")
    segments, cursor, output = [], 0., 0.
    for cut in snapped + [{"start_s": duration, "end_s": duration}]:
        if cut["start_s"] > cursor + TIME_TOLERANCE_S:
            length = cut["start_s"] - cursor
            segments.append({"segment_id": len(segments), "source_start_s": cursor,
                             "source_end_s": cut["start_s"], "output_start_s": output, "output_end_s": output + length})
            output += length
        cursor = cut["end_s"]
    expected_times = rec.frame_times[indices].copy()
    for cut in snapped:
        expected_times -= (cut["end_s"] - cut["start_s"]) * (rec.frame_times[indices] >= cut["end_s"] - TIME_TOLERANCE_S)
    return {"eligible_for_training": eligible, "review_status": card["status"],
            "requested_cuts": requested, "applied_cuts": snapped, "kept_segments": segments,
            "source_frame_indices": indices.tolist(), "expected_frame_times": expected_times.tolist(),
            "output_duration_s": output, "privacy_regions": regions, "warnings": warnings}


def retime_sensors(imu: pd.DataFrame | None, segments: list[dict]) -> pd.DataFrame | None:
    if imu is None:
        return None
    parts = []
    for segment in segments:
        part = imu[(imu.t >= segment["source_start_s"]) & (imu.t < segment["source_end_s"])].copy()
        part["source_t"] = part.t
        part["segment_id"] = segment["segment_id"]
        part["t"] = part.source_t - segment["source_start_s"] + segment["output_start_s"]
        parts.append(part)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(result) and not np.all(np.diff(result.t) > 0):
        raise ValueError("Retimed sensor timestamps must remain strictly increasing.")
    return result
