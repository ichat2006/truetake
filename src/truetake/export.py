"""Export reviewed derivatives, source-time maps and optional retimed IMU data."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from truetake.edit_plan import build_edit_plan, retime_sensors, TIME_TOLERANCE_S
from truetake.ingest import load_recording
from truetake.models import Recording


EXPORT_VERSION = 2
VIDEO_ENCODER = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_MOVFLAGS = "+faststart"
ENCODER_THREADS = 2
FILTER_THREADS = 1
CRF = 18
BLUR_RADIUS = 24
TIMEOUT_S = 300
MAX_TIME_ERROR_S = 0.0001
TIMEBASE = 1000000


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_root(path: str) -> Path:
    root = Path(path).resolve()
    protected = (Path(__file__).resolve().parents[2] / "data").resolve()
    if root == protected or protected in root.parents:
        raise ValueError("Exports must be written outside data/.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def omitted_entry(card: dict) -> dict | None:
    from truetake.licenses import may_export
    if not may_export(card):
        return {'clip_id':card['clip_id'],'export_status':'not_exported','eligible_for_training':False,
                'review_status':card['status'],'reason':'Licence does not permit modified media exports'}
    reason = "Controlled demo fixture is not independent training footage" if card.get("fixture_kind") else (
        "Rejected source clip is not repaired into an accepted task" if card["status"] == "REJECT" else None)
    if reason:
        return {"clip_id": card["clip_id"], "export_status": "not_exported", "eligible_for_training": False,
                "review_status": card["status"], "reason": reason}
    return None


def video_filters(plan: dict) -> str:
    filters = ["[0:v:0]settb=AVTB,setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2[v0]"]
    for index, region in enumerate(plan["privacy_regions"]):
        enabled = f"gte(t,{max(0., region['start_s'] - TIME_TOLERANCE_S):.9f})*lt(t,{region['end_s'] + TIME_TOLERANCE_S:.9f})"
        if region["full_frame_fallback"]:
            radius = max(1, min(BLUR_RADIUS, region["width"] // 4, region["height"] // 4))
            filters.append(f"[v{index}]boxblur=luma_radius={radius}:luma_power=3:chroma_radius={radius//2}:enable='{enabled}'[v{index+1}]")
        else:
            x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
            radius = max(1, min(BLUR_RADIUS, width // 4, height // 4))
            filters.extend([f"[v{index}]split=2[base{index}][roi{index}]",
                f"[roi{index}]crop={width}:{height}:{x}:{y},boxblur=luma_radius={radius}:luma_power=3:chroma_radius={max(1,radius//2)}[blur{index}]",
                f"[base{index}][blur{index}]overlay={x}:{y}:enable='{enabled}'[v{index+1}]"])
    selection = "+".join(f"gte(n,{cut['first_frame']})*lt(n,{cut['end_frame']})" for cut in plan["applied_cuts"]) or "0"
    offset = "+".join(f"{cut['end_s']-cut['start_s']:.9f}*gte(T,{max(0.,cut['end_s']-TIME_TOLERANCE_S):.9f})" for cut in plan["applied_cuts"]) or "0"
    filters.append(f"[v{len(plan['privacy_regions'])}]select='not({selection})',setpts='PTS-({offset})/TB'[out]")
    return ";".join(filters)


def probe_browser_video(path: Path) -> dict:
    """Check the actual output codec; encoder arguments alone are not evidence."""
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,pix_fmt,codec_tag_string", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True, timeout=TIMEOUT_S)
    streams = json.loads(result.stdout).get("streams", [])
    if (not streams or streams[0].get("codec_name") != "h264"
            or streams[0].get("pix_fmt") != VIDEO_PIXEL_FORMAT):
        raise ValueError("Export must contain browser-compatible H.264 video with yuv420p pixels.")
    return streams[0]


def export_clip(rec: Recording, card: dict, directory: str, table: pd.DataFrame | None = None,
                force: bool = False) -> dict:
    root = output_root(directory)
    omitted = omitted_entry(card)
    if omitted:
        return omitted
    clip_id = card["clip_id"]
    if Path(clip_id).name != clip_id or clip_id in ("", ".", ".."):
        raise ValueError("Clip ID must be a single path component.")
    plan = build_edit_plan(rec, card, table)
    from truetake.licenses import apply_license
    licensed=apply_license(card)
    if not licensed['eligibility']['eligible']:
        plan['eligible_for_training']=False
    source_hash = file_hash(Path(rec.video_path))
    sensor_hash = hashlib.sha256(rec.imu.to_csv(index=False, float_format="%.17g").encode()).hexdigest() if rec.imu is not None else None
    signature = {"version": EXPORT_VERSION, "source_sha256": source_hash, "sensor_sha256": sensor_hash,
                 "plan": plan, "review": card.get("claude_review"), "crf": CRF,
                 "encoding": {"encoder": VIDEO_ENCODER, "pix_fmt": VIDEO_PIXEL_FORMAT, "movflags": VIDEO_MOVFLAGS},
                 "code": [file_hash(Path(__file__)), file_hash(Path(__file__).with_name("edit_plan.py"))]}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()
    bucket = "accepted" if plan["eligible_for_training"] else "pending"
    folder = root / bucket / clip_id / fingerprint[:16]
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if cached.get("fingerprint") == fingerprint and all((folder/name).is_file() and file_hash(folder/name) == digest for name,digest in cached["artifact_sha256"].items()):
            return {**cached, "cached": True}
    video_name, frame_name, sensor_name = f"{clip_id}.mp4", f"{clip_id}_frame_map.csv", f"{clip_id}_sensors.csv"
    if (folder / video_name).resolve() == Path(rec.video_path).resolve():
        raise ValueError("An export must not replace its source video.")
    indices = np.array(plan["source_frame_indices"], dtype=int)
    expected = np.array(plan["expected_frame_times"])
    sensors = retime_sensors(rec.imu, plan["kept_segments"])
    with tempfile.TemporaryDirectory(prefix=".encoding-", dir=folder) as temporary:
        stage = Path(temporary)
        # setts preserves the final retained frame's original hold duration,
        # including variable-rate footage. No duplicate frame is introduced.
        tail = plan["output_duration_s"] - expected[-1]
        duration_filter = f"setts=duration='if(eq(N,{len(indices)-1}),{tail:.9f}/TB,DURATION)'"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", str(ENCODER_THREADS), "-i", rec.video_path,
            "-filter_complex_threads", str(FILTER_THREADS), "-filter_complex", video_filters(plan), "-map", "[out]", "-an",
            "-map_metadata", "-1", "-metadata:s:v:0", "rotate=0", "-c:v", VIDEO_ENCODER, "-threads", str(ENCODER_THREADS),
            "-preset", "veryfast", "-crf", str(CRF), "-bf", "0", "-pix_fmt", VIDEO_PIXEL_FORMAT, "-fps_mode", "passthrough",
            "-enc_time_base", f"1:{TIMEBASE}", "-video_track_timescale", str(TIMEBASE), "-bsf:v", duration_filter,
            "-movflags", VIDEO_MOVFLAGS, str(stage / video_name)], capture_output=True, text=True, check=True, timeout=TIMEOUT_S)
        encoding = probe_browser_video(stage / video_name)
        encoded = load_recording(str(stage / video_name))
        if len(encoded.frame_times) != len(expected) or not np.allclose(encoded.frame_times, expected, atol=MAX_TIME_ERROR_S, rtol=0):
            raise ValueError("Encoded video did not preserve the planned frame timestamps.")
        if abs(encoded.meta["duration_s"] - plan["output_duration_s"]) > MAX_TIME_ERROR_S:
            raise ValueError("Encoded video duration differs from the edit map.")
        expected_size = tuple(int(np.ceil(rec.meta[key] / 2) * 2) for key in ("display_width", "display_height"))
        if (encoded.meta["display_width"], encoded.meta["display_height"]) != expected_size or encoded.meta["rotation"] != 0:
            raise ValueError("Encoded display orientation or dimensions changed unexpectedly.")
        segment_ids = np.zeros(len(expected), dtype=int)
        for segment in plan["kept_segments"]:
            inside = (rec.frame_times[indices] >= segment["source_start_s"] - TIME_TOLERANCE_S) & (rec.frame_times[indices] < segment["source_end_s"] - TIME_TOLERANCE_S)
            segment_ids[inside] = segment["segment_id"]
        pd.DataFrame({"output_frame": np.arange(len(indices)), "output_t": encoded.frame_times,
                      "source_frame": indices, "source_t": rec.frame_times[indices], "segment_id": segment_ids}).to_csv(stage / frame_name, index=False, float_format="%.17g")
        artifacts = [video_name, frame_name]
        if sensors is not None and len(sensors):
            sensors.to_csv(stage / sensor_name, index=False, float_format="%.17g")
            artifacts.append(sensor_name)
        result = {"schema_version": EXPORT_VERSION, "clip_id": clip_id, "fingerprint": fingerprint,
            "export_status": "exported" if plan["eligible_for_training"] else "retained_pending",
            "eligible_for_training": plan["eligible_for_training"], "cached": False,
            "source_video_path": str(Path(rec.video_path).resolve()), "source_sha256": source_hash,
            "source_card_path": card.get("card_path"),
            "source_review_sha256": hashlib.sha256(json.dumps(card.get("claude_review"), sort_keys=True).encode()).hexdigest(),
            "review_model": (card.get("claude_review") or {}).get("provenance", {}).get("model"),
            "operator_request": card.get("operator_request", ""),
            "source_duration_s": rec.meta["duration_s"], "output_duration_s": encoded.meta["duration_s"],
            "video_path": str(folder/video_name), "frame_map_path": str(folder/frame_name),
            "sensors_path": str(folder/sensor_name) if sensor_name in artifacts else None,
            "manifest_path": str(manifest_path), "source_rotation_deg": rec.meta["rotation"], "output_rotation_deg": 0,
            "plan": {key:value for key,value in plan.items() if key not in ("source_frame_indices", "expected_frame_times")},
            "source_frame_count": len(rec.frame_times), "output_frame_count": len(indices),
            "video_encoding": encoding,
            "audio_policy": "Silent derivative; audio remains in the original recording",
            "sensor_policy": "Original rad/s and m/s^2 device axes preserved. Timestamps retimed only; original clock offset/calibration is unchanged. Do not integrate across segment joins.",
            "sensor_status": "no motion data" if rec.imu is None else "retimed" if sensor_name in artifacts else "no samples in retained video",
            "sensor_sample_count": len(sensors) if sensors is not None else 0,
            "sensor_join_times_s": [segment["output_start_s"] for segment in plan["kept_segments"]][1:],
            "edited_derivative": True, "source_verification_not_transferable": True,
            "post_edit_task_assessment": "Not rerun; source review and frame map retained for inspection",
            "artifact_sha256": {name:file_hash(stage/name) for name in artifacts}}
        for name in artifacts:
            (stage/name).replace(folder/name)
        manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def write_collection_manifest(entries: dict[str, dict], directory: str) -> dict:
    root = output_root(directory)
    result = {"schema_version": EXPORT_VERSION, "entries": entries,
              "accepted_count": sum(entry.get("eligible_for_training", False) for entry in entries.values()),
              "pending_count": sum(entry["export_status"] == "retained_pending" for entry in entries.values()),
              "omitted_count": sum(entry["export_status"] == "not_exported" for entry in entries.values()),
              "errors": [name for name,entry in entries.items() if entry["export_status"] == "failed"],
              "manifest_path": str(root / "collection.json")}
    Path(result["manifest_path"]).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def export_collection(batch_path: str, directory: str, rerun: bool = False, force: bool = False) -> dict:
    from truetake.report import write_batch_report
    from truetake.viz_rerun import save_recording, save_batch_map
    batch = json.loads(Path(batch_path).read_text())
    entries = {}
    for name, card in batch["cards"].items():
        rec, table = None, None
        try:
            entry = omitted_entry(card)
            if entry is None or rerun:
                video = Path(card["video_path"])
                sensor = Path(card.get("sensors_path") or video.with_name(video.stem + "_sensors.csv"))
                rec = replace(load_recording(str(video), str(sensor) if card.get("sensor_present") else None), clip_id=name)
                table = pd.DataFrame(json.loads(Path(card["table_path"]).read_text()))
            if entry is None:
                entry = export_clip(rec, card, directory, table, force=force)
            entries[name] = card["export"] = entry
            if rerun:
                save_recording(rec, table, card, card["rerun_path"])
            print(f"{name}: {entry['export_status']}", flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            entries[name] = card["export"] = {"clip_id": name, "export_status": "failed", "eligible_for_training": False,
                                             "reason": f"{type(error).__name__}: {error}"}
            print(f"{name}: export failed: {error}", flush=True)
    collection = write_collection_manifest(entries, directory)
    batch["analysis"]["exports"] = {key:value for key,value in collection.items() if key != "entries"}
    if rerun:
        save_batch_map(batch["analysis"], str(Path(batch_path).parent / "batch_map.rrd"))
    write_batch_report(batch["cards"], batch["analysis"], str(Path(batch_path).parent))
    return collection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export corrected clips from a saved TrueTake batch without model or API calls.")
    parser.add_argument("batch")
    parser.add_argument("--output-dir", default="outputs/demo/exports")
    parser.add_argument("--rerun", action="store_true", help="Refresh source Rerun recordings with edit-map annotations")
    parser.add_argument("--force", action="store_true", help="Re-encode videos even when a matching export cache exists")
    args = parser.parse_args(argv)
    try:
        result = export_collection(args.batch, args.output_dir, args.rerun, args.force)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"truetake export: {error}\n")
    print(f"Collection: {result['manifest_path']}")
    return 2 if result["errors"] else 0
