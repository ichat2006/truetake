"""Save an inspectable clip timeline without starting a GPU viewer."""

import json
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from truetake.detect_basic import ANALYSIS_WIDTH, iter_sampled_frames
from truetake.models import Recording


DISPLAY_WIDTH = 960
JPEG_QUALITY = 90
PRIVACY_BLUR_KERNEL = 51
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def save_batch_map(analysis: dict, path: str) -> None:
    stream = rr.RecordingStream("TrueTake batch")
    stream.save(path)
    try:
        stream.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial2DView(origin="/clips/map", name="Clip map"),
            rrb.TextDocumentView(origin="/review/batch", name="Batch evidence")), collapse_panels=True))
        points = analysis["map"]
        if points:
            colors = {"ACCEPT": [70, 220, 150], "REJECT": [250, 110, 100], "REVIEW": [245, 200, 95]}
            stream.log("clips/map", rr.Points2D([[point["x"], point["y"]] for point in points],
                        labels=[point["clip_id"] for point in points],
                        colors=[colors[point["status"]] for point in points]), static=True)
        stream.log("review/batch", rr.TextDocument(json.dumps(analysis, indent=2)), static=True)
        stream.flush(timeout_sec=30)
    finally:
        stream.disconnect()


def save_recording(rec: Recording, table: pd.DataFrame, card: dict, path: str) -> str:
    stream = rr.RecordingStream("TrueTake", recording_id=rec.clip_id)
    stream.save(path)
    try:
        stream.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial2DView(origin="/video/image", name="Clip and hands"),
            rrb.Vertical(rrb.TextDocumentView(origin="/review/card", name="Review"),
                         rrb.TimeSeriesView(origin="/signals", name="Measured signals")),
            column_shares=[2, 1]), rrb.TimePanel(timeline="clip", state="expanded"), collapse_panels=True))
        lines = [f"# {rec.clip_id}", f"**Status: {card.get('status', 'REVIEW')}**",
                 f"Contract score: {card.get('quality_score', 'pending')} / 100 (not an acceptance probability)",
                 f"Estimated usable: {card.get('usable_seconds', 0):.1f} s",
                 card.get("usable_evidence_kind", ""),
                 "Decision: " + "; ".join(card.get("decision_reasons", [])),
                 f"AI evidence: {(card.get('claude_usage') or {}).get('mode', 'baseline')}; {(card.get('claude_usage') or {}).get('initial_images', 'unknown')} initial images",
                 f"Local detectors: {(card.get('detector_stats') or {}).get('mode', 'multipass')}; {(card.get('detector_stats') or {}).get('analysis_decode_passes', 'unknown')} analysis decode pass(es)",
                 f"Claude: {card.get('claude_status', 'pending')}", card.get("fixture_kind", ""), "## Checks"]
        for check in card.get("checks", []):
            verdict = "PASS" if check["passed"] is True else "FAIL" if check["passed"] is False else "UNKNOWN"
            lines.append(f"- **{verdict} {check['name']}**: {check['reason']}")
        exported = card.get("export") or {}
        if exported:
            lines.extend(["## Corrected output", f"{exported['export_status']}; training eligible: {exported.get('eligible_for_training', False)}",
                          exported.get("video_path", exported.get("reason", "")),
                          "This timeline uses original times. Edit markers map them to the exported video."])
        stream.log("review/card", rr.TextDocument("\n\n".join(lines), media_type="text/markdown"), static=True)
        for row, frame in iter_sampled_frames(rec, table, max_width=DISPLAY_WIDTH):
            stream.set_time("clip", duration=float(table.at[row, "t"]))
            if "claude_selected" in table and table.at[row, "claude_selected"]:
                stream.log("review/claude_evidence", rr.TextLog(f"Frame {row}: {table.at[row, 'claude_selection_reason']}"))
            if any(a <= float(table.at[row, "t"]) <= b for a, b in card.get("privacy_blur_intervals", [])):
                # Preserve the previous blur strength relative to image width.
                kernel = round(PRIVACY_BLUR_KERNEL * frame.shape[1] / ANALYSIS_WIDTH) | 1
                frame = cv2.GaussianBlur(frame, (kernel, kernel), 0)
            stream.log("video/image", rr.Image(frame, color_model="BGR").compress(jpeg_quality=JPEG_QUALITY))
            height, width = frame.shape[:2]
            hands = table.at[row, "hand_landmarks"] if "hand_landmarks" in table else None
            if hands is not None:
                strips = []
                for hand in hands:
                    points = np.array(hand["points"])[:, :2] * [width, height]
                    strips.extend([points[[a, b]] for a, b in HAND_CONNECTIONS])
                stream.log("video/image/hands", rr.LineStrips2D(strips, colors=[70, 230, 170], radii=1.0))
            for column in ("brightness", "blur_var", "frame_change", "hand_count", "hands_inside",
                           "face_confidence", "flow_pan", "flow_tilt", "flow_roll",
                           "gx", "gy", "gz", "ax", "ay", "az"):
                if column in table and pd.notna(table.at[row, column]):
                    stream.log(f"signals/{column}", rr.Scalars(float(table.at[row, column])))
        for check in card.get("checks", []):
            for flag in check["flags"]:
                start, end, note = flag if isinstance(flag, (tuple, list)) else (flag["start_s"], flag["end_s"], flag["note"])
                stream.set_time("clip", duration=start)
                stream.log("review/flags", rr.TextLog(f"{start:.2f}–{end:.2f}s: {note}"))
        for window in (card.get("evidence_plan") or {}).get("candidate_windows", []):
            stream.set_time("clip", duration=window["start_s"])
            stream.log("review/claude_candidates", rr.TextLog(f"{window['start_s']:.2f}–{window['end_s']:.2f}s: {window['kind']}; inspection hint, no automatic cut"))
        for segment in exported.get("plan", {}).get("kept_segments", []):
            stream.set_time("clip", duration=segment["source_start_s"])
            stream.log("review/edits", rr.TextLog(f"KEEP source {segment['source_start_s']:.3f}–{segment['source_end_s']:.3f}s → edited {segment['output_start_s']:.3f}–{segment['output_end_s']:.3f}s"))
        for cut in exported.get("plan", {}).get("applied_cuts", []):
            stream.set_time("clip", duration=cut["start_s"])
            stream.log("review/edits", rr.TextLog(f"CUT source {cut['start_s']:.3f}–{cut['end_s']:.3f}s; completion context retained"))
        stream.flush(timeout_sec=30)
    finally:
        stream.disconnect()
    return str(Path(path))
