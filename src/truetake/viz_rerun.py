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
NORMALIZATION_EPSILON = 1e-8
KEEP_COLOR = [70, 220, 150]
CUT_COLOR = [250, 110, 100]
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


def aligned_motion(table: pd.DataFrame, alignment: dict) -> dict:
    """Display mapped, lag-adjusted signals in standard deviations, not physical equivalence."""
    from truetake.verify_gyro import prepare_signals, _interpolate, VIDEO_AXES
    if not alignment.get('axis_mapping') or alignment.get('offset_s') is None:
        return {}
    if not {'flow_t','flow_status',*VIDEO_AXES} <= set(table):
        return {}
    grid,videos=prepare_signals(table)
    traces={}
    for axis in ['flow_pan','flow_roll']:
        pair=alignment['axis_mapping'][axis]
        if pair['axis'] not in table:continue
        gyro=_interpolate(table.t.to_numpy(),table[pair['axis']].to_numpy(),
                          grid+alignment['offset_s'])*pair['sign']
        signals=[]
        for values in [videos[VIDEO_AXES.index(axis)],gyro]:
            valid=np.isfinite(values)
            result=values.copy()
            if valid.any():
                result[valid]=(values[valid]-values[valid].mean())/max(values[valid].std(),NORMALIZATION_EPSILON)
            signals.append(result)
        traces[axis]=(grid,*signals)
    return traces


def segment_rows(duration: float, card: dict) -> list[dict]:
    """Only applied export cuts are red; compiled proposals are never called cuts."""
    cuts=(card.get('export') or {}).get('plan',{}).get('applied_cuts',[])
    rows=[];cursor=0.
    for cut in cuts:
        start,end=cut['start_s'],cut['end_s']
        if start>cursor:rows.append({'start':cursor,'end':start,'kind':'KEEP','reason':'Retained original footage'})
        reasons=[check['reason'] for check in card.get('checks',[]) for flag in check.get('flags',[])
                 if isinstance(flag,(list,tuple)) and flag[0]<end and flag[1]>start]
        rows.append({'start':start,'end':end,'kind':'CUT','reason':reasons[0] if reasons else 'Compiled unusable interval'})
        cursor=end
    if cursor<duration:rows.append({'start':cursor,'end':duration,'kind':'KEEP','reason':'Retained original footage'})
    return rows


def save_recording(rec: Recording, table: pd.DataFrame, card: dict, path: str) -> str:
    stream = rr.RecordingStream("TrueTake", recording_id=rec.clip_id)
    stream.save(path)
    try:
        stream.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial2DView(origin="/video/image", name="Clip and hands"),
            rrb.Vertical(rrb.TextDocumentView(origin="/review/card", name="Review"),
                         rrb.TimeSeriesView(origin="/signals/quality", name="Image quality"),
                         rrb.TimeSeriesView(origin="/signals/hands", name="Hands"),
                         rrb.TimeSeriesView(origin="/signals/motion", name="Motion"),
                         rrb.TimeSeriesView(origin="/aligned", name="Video / mapped gyro (standard deviations)"),
                         rrb.Spatial2DView(origin="/segments", name="Kept / cut — original seconds")),
            column_shares=[2, 1]), rrb.TimePanel(timeline="clip", state="expanded"), collapse_panels=True))
        from truetake.presentation import badge, plain_reason
        lines = [f"# {badge(card)} — {plain_reason(card)}", f"**{rec.clip_id}**",
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
            for group, columns in {'quality':['brightness','blur_var'], 'hands':['hand_count','hands_inside'],
                                   'motion':['frame_change']}.items():
                for column in columns:
                    if column in table and pd.notna(table.at[row,column]):
                        stream.log(f"signals/{group}/{column}",rr.Scalars(float(table.at[row,column])))
        for axis,(grid,video,gyro) in aligned_motion(table,card.get('gyro_alignment') or {}).items():
            stream.log(f'aligned/{axis}/video',rr.SeriesLines(colors=[100,196,255],names=f'{axis}: video'),static=True)
            stream.log(f'aligned/{axis}/gyro',rr.SeriesLines(colors=[244,199,119],names=f'{axis}: mapped gyro'),static=True)
            for t,v,g in zip(grid,video,gyro):
                stream.set_time('clip',duration=float(t))
                # Explicit NaNs break gaps rather than connecting missing motion.
                stream.log(f'aligned/{axis}/video',rr.Scalars(float(v)))
                stream.log(f'aligned/{axis}/gyro',rr.Scalars(float(g)))
        segments=segment_rows(float(rec.meta['duration_s']),card)
        stream.log('segments',rr.Boxes2D(
            centers=[[(part['start']+part['end'])/2,0] for part in segments],
            half_sizes=[[(part['end']-part['start'])/2,.5] for part in segments],
            colors=[KEEP_COLOR if part['kind']=='KEEP' else CUT_COLOR for part in segments],
            labels=[f"{part['kind']} {part['start']:.1f}–{part['end']:.1f}s: {part['reason']}" for part in segments],
            show_labels=True),static=True)
        for part in segments:
            stream.set_time('clip',duration=part['start'])
            stream.log('review/segments',rr.TextLog(f"{part['kind']} {part['start']:.1f}–{part['end']:.1f}s: {part['reason']}"))
        review=card.get('claude_review') or {}
        evidence={item['id']:item for item in review.get('evidence',[])}
        for step in review.get('steps',[]):
            for key in step.get('evidence_ids',[]):
                if key in evidence:
                    stream.set_time('clip',duration=float(evidence[key]['t']))
                    stream.log('review/steps',rr.TextLog(step['name']))
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
