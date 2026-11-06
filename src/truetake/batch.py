"""Sequential CPU demo pipeline with cached detectors and offline reports."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from truetake.ingest import load_recording
from truetake.frame_table import build_table
from truetake.detect_basic import detect_basic
from truetake.detect_people import detect_people
from truetake.detect_visual import detect_visual, load_visual_model
from truetake.detect_claude import detect_claude, MODEL
from truetake.models import CheckResult
from truetake.quality import check_quality
from truetake.report import build_card, write_json, write_batch_report
from truetake.value import analyze_batch
from truetake.viz_rerun import save_recording, save_batch_map
import yaml
from truetake.contracts import load_contract
from truetake.evidence import select_evidence, save_selected_frames
from truetake.compare import compare_reviews
from truetake.detect_onepass import detect_onepass
from truetake.detect_flow import detect_flow
from truetake.calibration import load_profile
from truetake.verify_gyro import measure_alignment, check_verification
from truetake.export import export_clip, write_collection_manifest
import subprocess


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = SOURCE_ROOT / "outputs/models"


def verify_recording(rec, table: pd.DataFrame, profile: dict | None) -> tuple[pd.DataFrame, list[CheckResult]]:
    """Run independently of task quality; cached detectors never supply a profile."""
    table = table.copy()
    meta = dict(table.iloc[0]["clip_meta"] or {})
    meta.pop("gyro_calibration", None)
    if profile is not None:
        if profile.get("source_rotation_deg") not in (None, rec.meta["rotation"]):
            raise ValueError("Gyro profile rotation differs from the recording.")
        meta["gyro_calibration"] = profile
    table.at[table.index[0], "clip_meta"] = meta
    if rec.imu is None:
        return table, [CheckResult("verification.gyro", None, 0.0, "no motion data", [])]
    if profile is None or profile["status"] != "calibrated":
        return table, [CheckResult("verification.gyro", None, 0.0, "calibrated gyro profile required", [])]
    table = measure_alignment(detect_flow(rec, table))
    return table, check_verification(table)


def apply_verification_quality(quality: dict, checks: list[CheckResult]) -> dict:
    """A confirmed sensor mismatch blocks export, even when task quality passes."""
    failures = [check.reason for check in checks if check.passed is False]
    if not failures:
        return quality
    return {**quality, "status": "REJECT", "usable_intervals": [], "usable_seconds": 0.0,
            "approved_usable_seconds": 0.0,
            "decision_reasons": list(quality.get("decision_reasons", [])) + failures}


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def run_batch(input_dir: str, contract_path: str, output_dir: str, include_duplicate: bool = False,
              model: str = MODEL, instructions: str = "", review_mode: str = "focused", compare_to: str | None = None,
              detector_mode: str = "multipass", export: bool = False, gyro_profile: str | None = None,
              clip_ids: list[str] | None = None, render: bool = True, media_metadata: dict | None = None,
              progress=None) -> dict:
    output = Path(output_dir).resolve()
    protected = (SOURCE_ROOT / "data").resolve()
    if output == protected or protected in output.parents:
        raise ValueError("Generated reports must be saved outside data/.")
    contract_file = Path(contract_path)
    contract = load_contract(str(contract_file), instructions)
    from truetake.licenses import catalog_metadata
    if media_metadata is None:media_metadata=catalog_metadata()
    profile = load_profile(gyro_profile) if gyro_profile else None
    baseline = json.loads(Path(compare_to).read_text()) if compare_to else None
    if review_mode not in ("baseline", "focused"):
        raise ValueError("Review mode must be baseline or focused.")
    if contract.get("schema_version", 1) != 2:
        review_mode = "baseline"
    if detector_mode not in ("multipass", "onepass"):
        raise ValueError("Detector mode must be multipass or onepass.")
    if detector_mode == "onepass" and review_mode != "focused":
        raise ValueError("One-pass detectors currently require focused review; use multipass for baseline review.")
    videos = sorted(Path(input_dir).rglob("*.mp4"))
    if clip_ids is not None:
        videos = [video for video in videos if video.stem in set(clip_ids)]
    if not videos:
        raise ValueError("No MP4 clips found in input directory.")
    ids = [video.stem for video in videos]
    if len(set(ids)) != len(ids):
        raise ValueError("Clip filenames must have unique stems across the batch.")
    output.mkdir(parents=True, exist_ok=True)
    detector_sources = [Path(__file__).with_name(name) for name in
                        ("frame_table.py", "ingest.py", "detect_basic.py", "detect_people.py", "detect_visual.py")]
    if detector_mode == "onepass":
        detector_sources += [Path(__file__).with_name(name) for name in ("detect_onepass.py", "evidence.py")]
    models = [MODEL_DIR / name for name in ("ViT-B-32.pt", "hand_landmarker.task", "face_detector.tflite")]
    detector_version = detector_mode + ":" + fingerprint(detector_sources + models + [SOURCE_ROOT / "uv.lock"])
    tables, recordings, cards, errors = {}, {}, {}, []
    visual_model = None
    started = time.monotonic()
    for video in videos:
        clip_id = video.stem
        clip_dir = output / clip_id
        clip_dir.mkdir(exist_ok=True)
        sensors = video.with_name(video.stem + "_sensors.csv")
        try:
            rec = load_recording(str(video), str(sensors) if sensors.is_file() else None)
            key = detector_version + fingerprint([video] + ([sensors] if sensors.is_file() else []))
            cache_name = "detectors.onepass" if detector_mode == "onepass" else "detectors"
            cache = clip_dir / f"{cache_name}.json"
            cache_key = clip_dir / f"{cache_name}.sha256"
            table = None
            if cache.is_file() and cache_key.is_file() and cache_key.read_text() == key:
                candidate = pd.DataFrame(json.loads(cache.read_text()))
                if all(Path(path).is_file() for path in candidate.thumbnail_path.dropna()):
                    table = candidate
            cached = table is not None
            if table is None:
                detector_started, model_load_s = time.monotonic(), 0.
                if detector_mode == "multipass":
                    table = detect_people(rec, detect_basic(rec, build_table(rec, contract)), str(MODEL_DIR))
                if visual_model is None:
                    model_started = time.monotonic()
                    visual_model = load_visual_model(str(models[0]))
                    model_load_s = time.monotonic() - model_started
                if detector_mode == "onepass":
                    table = detect_onepass(rec, build_table(rec, contract), visual_model, str(clip_dir), str(MODEL_DIR))
                else:
                    table = detect_visual(rec, table, visual_model, str(clip_dir))
                    table["detector_stats"] = pd.Series([None] * len(table), index=table.index, dtype=object)
                    table.at[0, "detector_stats"] = {"mode": "multipass", "analysis_decode_passes": 3,
                        "evidence_decode_passes": 1 if review_mode == "focused" else 0,
                        "source_frames_grabbed": 3 * len(rec.frame_times), "sampled_frames": len(table),
                        "hand_frames": int((table.people_status == "ok").sum()), "face_frames": int((table.face_status == "ok").sum()),
                        "clip_frames": int((table.clip_status == "ok").sum()),
                        "multipass_clip_frames_at_2hz": int((table.clip_status == "ok").sum()),
                        "hash_frames": int(table.phash.notna().sum()),
                        "detector_elapsed_s": round(time.monotonic() - detector_started - model_load_s, 3)}
                cache.write_text(table.to_json(orient="records", double_precision=15), encoding="utf-8")
                cache_key.write_text(key)
            table.at[table.index[0], "contract"] = contract
            # Verification has its own profile on every run and also runs for
            # clips that Claude or the task checks subsequently reject.
            table, verification = verify_recording(rec, table, profile)
            prompt = "review_v2.md" if contract.get("schema_version", 1) == 2 else "review_v1.md"
            if review_mode == "focused":
                if detector_mode == "multipass":
                    table = save_selected_frames(rec, select_evidence(table), str(clip_dir))
                prompt = "review_focused_v1.md"
            table = detect_claude(table, str(output / "claude_cache"), str(SOURCE_ROOT / "prompts" / prompt), model,
                                  rec=rec, evidence_dir=str(clip_dir))
            checks, quality = check_quality(table)
            checks.extend(verification)
            quality = apply_verification_quality(quality, verification)
            from truetake.verdicts import apply_verdict
            checks, quality = apply_verdict(table, checks, quality)
            card = build_card(clip_id, table, checks, quality)
            from truetake.licenses import apply_license
            card=apply_license({**card,**(media_metadata or {}).get(clip_id,{})},contract)
            card["gyro_alignment"] = table.iloc[0].get("gyro_alignment")
            card.update(video_path=str(video.resolve()), card_path=str(clip_dir / "card.json"),
                        sensors_path=str(sensors.resolve()) if rec.imu is not None else None,
                        table_path=str(clip_dir / "frames.json"), rerun_path=str(clip_dir / "timeline.rrd"),
                        detector_cache_hit=cached, detector_fingerprint=detector_version)
            tables[clip_id], recordings[clip_id], cards[clip_id] = table, rec, card
            print(f"{clip_id}: {quality['status']}; quality {quality['quality_score']:.1f}; AI {card['claude_status']}; cached={cached}", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append({"clip_id": clip_id, "error": str(error)})
            print(f"{clip_id}: processing failed: {error}", flush=True)
    if include_duplicate and tables:
        source = next(iter(tables))
        duplicate = source + "_controlled_reupload"
        tables[duplicate] = tables[source].copy()
        recordings[duplicate] = replace(recordings[source], clip_id=duplicate)
        cards[duplicate] = {**cards[source], "clip_id": duplicate,
                            "fixture_kind": f"CONTROLLED DEMO FIXTURE: same video as {source}; not an independent recording"}
        destination = output / duplicate
        destination.mkdir(exist_ok=True)
        cards[duplicate].update(card_path=str(destination / "card.json"), table_path=str(destination / "frames.json"),
                                rerun_path=str(destination / "timeline.rrd"))
        cards[duplicate]["claude_usage"] = {**(cards[source].get("claude_usage") or {}),
            "actual_api_calls": 0, "actual_images_sent": 0, "actual_input_tokens": 0, "actual_output_tokens": 0,
            "reuse_reason": "controlled duplicate reuses source review"}
    analysis = analyze_batch(tables, cards)
    original_stats = [card.get("detector_stats") or {} for card in cards.values() if not card.get("fixture_kind")]
    analysis["detector_totals"] = {key: sum(stats.get(key, 0) for stats in original_stats) for key in
        ("source_frames_grabbed", "sampled_frames", "hand_frames", "face_frames", "clip_frames", "hash_frames",
         "hands_skipped_dark", "clip_skipped_dark", "multipass_clip_frames_at_2hz")}
    if baseline is not None:
        analysis["evidence_comparison"] = compare_reviews(baseline, cards)
        analysis["evidence_comparison"]["scope"] = f"{detector_mode} detectors with {review_mode} review; development comparison"
    analysis.update(errors=errors, elapsed_s=round(time.monotonic() - started, 2),
                    review_mode=review_mode,
                    detector_mode=detector_mode,
                    evaluation="Human goal/quality labels are required before reporting accuracy.",
                    calibration={"status": profile["status"] if profile else "uncalibrated",
                                 "profile": gyro_profile,
                                 "train_clip_ids": profile.get("train_clip_ids", []) if profile else []})
    if export:
        entries = {}
        for clip_id, card in cards.items():
            try:
                entries[clip_id] = card["export"] = export_clip(recordings[clip_id], card, str(output / "exports"), tables[clip_id])
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                entries[clip_id] = card["export"] = {"clip_id": clip_id, "export_status": "failed", "eligible_for_training": False,
                                                    "reason": str(error)}
                errors.append({"clip_id": clip_id, "stage": "export", "error": str(error)})
        collection = write_collection_manifest(entries, str(output / "exports"))
        analysis["exports"] = {key:value for key,value in collection.items() if key != "entries"}
    for clip_id, card in cards.items():
        card["scene_distinctness"] = analysis["scene_distinctness"].get(clip_id)
        tables[clip_id].to_json(card["table_path"], orient="records", indent=2, double_precision=15)
        write_json(card["card_path"], card)
        if render and card['license_permissions']['modify']:
            save_recording(recordings[clip_id], tables[clip_id], card, card["rerun_path"])
        elif card['license_permissions']['modify'] and not Path(card["rerun_path"]).is_file():
            temporary = Path(card["rerun_path"]).with_name(".timeline.recording.rrd")
            save_recording(recordings[clip_id], tables[clip_id], card, str(temporary))
            temporary.replace(card["rerun_path"])
    report = None
    if render:
        save_batch_map(analysis, str(output / "batch_map.rrd"))
        report = write_batch_report(cards, analysis, str(output))
        print(f"Report: {report}", flush=True)
    return {"report": report, "cards": cards, "analysis": analysis}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TrueTake's CPU review demo and Rerun recordings.")
    parser.add_argument("input_dir")
    parser.add_argument("--contract", default="contracts/mercor_folding.yaml")
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--include-duplicate", action="store_true", help="Include a clearly labeled controlled reupload fixture")
    parser.add_argument("--claude-model", default=MODEL)
    parser.add_argument("--instructions", default="", help="Operator collection request, e.g. explicitly allow partial demonstrations")
    parser.add_argument("--review-mode", choices=["focused", "baseline"], default="focused",
                        help="Focused: selected task evidence; baseline: 12 evenly spaced images")
    parser.add_argument("--compare-to", help="Saved baseline batch.json for verdict and original token-usage comparison")
    parser.add_argument("--detector-mode", choices=["multipass", "onepass"], default="multipass",
                        help="Onepass shares a single decode across local tiers and saves selected Claude frames in that pass")
    parser.add_argument("--export", action="store_true", help="Write corrected videos, edit maps, and retimed sensor CSVs")
    parser.add_argument("--gyro-profile", help="Version-1 calibrated profile with a threshold and train_clip_ids")
    args = parser.parse_args(argv)
    try:
        result = run_batch(args.input_dir, args.contract, args.output_dir, args.include_duplicate, args.claude_model, args.instructions, args.review_mode, args.compare_to, args.detector_mode, args.export, args.gyro_profile)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"truetake batch: {error}\n")
    return 2 if result["analysis"]["errors"] else 0
