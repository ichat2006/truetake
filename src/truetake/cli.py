"""The initial CLI loads inputs; quality and verification will follow later."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from truetake.ingest import load_recording
from truetake.frame_table import build_table
from truetake.detect_basic import detect_basic
from truetake.detect_flow import detect_flow
from truetake.verify_gyro import measure_alignment, check_verification
from truetake.calibration import diagnose_alignment, load_profile
from truetake.detect_people import detect_people
from truetake.contracts import load_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load a TrueTake recording and inspect its timing and sensor metadata."
    )
    parser.add_argument("video", help="Video file to load")
    parser.add_argument("--sensors", help="Normalized CSV: t,gx,gy,gz,ax,ay,az")
    parser.add_argument("--contract", help="Task contract YAML (loaded, not evaluated yet)")
    parser.add_argument("--basic", action="store_true", help="Measure brightness, blur, and frame change at 10 Hz")
    parser.add_argument("--people", action="store_true", help="Add CPU hand and face detection")
    parser.add_argument("--rerun", action="store_true", help="Save a Rerun timeline (includes --people)")
    parser.add_argument("--flow", action="store_true", help="Also estimate camera motion (includes --basic)")
    parser.add_argument("--verify", action="store_true", help="Measure gyro alignment (includes --flow; diagnostic until calibrated)")
    parser.add_argument("--diagnose-gyro", action="store_true", help="Plot alignment and validate a first-segment mapping on later windows")
    parser.add_argument("--gyro-profile", help="Provisional mapping or calibrated JSON with threshold and train_clip_ids")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated tables (default: outputs)")
    args = parser.parse_args(argv)
    args.verify = args.verify or args.diagnose_gyro or args.gyro_profile is not None

    try:
        contract = None
        if args.contract:
            contract_path = Path(args.contract)
            if contract_path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError("Task contract must be a YAML file.")
            contract = load_contract(str(contract_path))
            if not isinstance(contract, dict) or not isinstance(contract.get("task"), str):
                raise ValueError("Task contract must be a mapping with a string 'task'.")
            if not contract["task"].strip():
                raise ValueError("Task contract 'task' must not be empty.")
            if contract.get("draft"):
                raise ValueError("Draft contract is for review; its rule schema is not implemented yet.")
        rec = load_recording(args.video, args.sensors)
        if args.gyro_profile:
            profile = load_profile(args.gyro_profile)
            if profile.get("source_rotation_deg") not in (None, rec.meta["rotation"]):
                raise ValueError("Gyro profile rotation differs from the recording.")
            rec.meta["gyro_calibration"] = profile
        table_path = None
        table = None
        verification = None
        diagnostic_paths = None
        if args.basic or args.flow or args.verify or args.people or args.rerun:
            output_dir = Path(args.output_dir).resolve()
            data_dir = (Path(__file__).resolve().parents[2] / "data").resolve()
            if output_dir == data_dir or data_dir in output_dir.parents:
                raise ValueError("Generated tables must be saved outside data/.")
            table = detect_basic(rec, build_table(rec, contract))
            if args.people or args.rerun:
                table = detect_people(rec, table, "outputs/models")
            if args.flow or args.verify:
                table = detect_flow(rec, table)
            if args.verify:
                table = measure_alignment(table)
                verification = asdict(check_verification(table)[0])
            if args.diagnose_gyro:
                table = diagnose_alignment(table, rec.clip_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            table_path = output_dir / f"{rec.clip_id}_frames.json"
            table_path.write_text(table.to_json(orient="records", indent=2, double_precision=15), encoding="utf-8")
            if args.rerun:
                from truetake.viz_rerun import save_recording
                save_recording(rec, table, {"clip_id": rec.clip_id, "status": "REVIEW", "reason": "quality checks pending"},
                               str(output_dir / f"{rec.clip_id}.rrd"))
            if args.diagnose_gyro:
                from truetake.viz_alignment import plot_alignment

                diagnostics = table.iloc[0]["gyro_diagnostics"]
                report_path = output_dir / f"{rec.clip_id}_alignment.json"
                plot_path = output_dir / f"{rec.clip_id}_alignment.png"
                report_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
                plot_alignment(table, str(plot_path))
                profile_path = None
                if diagnostics["profile"] is not None:
                    profile_path = output_dir / f"{rec.clip_id}_gyro_candidate.json"
                    profile_path.write_text(json.dumps(diagnostics["profile"], indent=2, allow_nan=False), encoding="utf-8")
                diagnostic_paths = {"report": str(report_path), "plot": str(plot_path),
                                    "candidate_profile": str(profile_path) if profile_path else None,
                                    "summary": diagnostics["summary"]}
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(2, f"truetake: {error}\n")

    sensor_rate = None
    if rec.imu is not None and len(rec.imu) > 1:
        sensor_rate = float(1 / np.median(np.diff(rec.imu["t"])))
    summary = {
        "clip_id": rec.clip_id,
        "stage": "verify" if args.verify else "flow" if args.flow else "people" if args.people or args.rerun else "basic" if args.basic else "ingest",
        "review_status": "not_run",
        "task": None if contract is None else contract["task"],
        "frame_count": len(rec.frame_times),
        "duration_s": rec.meta["duration_s"],
        "duration_estimated": rec.meta["duration_estimated"],
        "width": rec.meta["width"],
        "height": rec.meta["height"],
        "display_width": rec.meta["display_width"],
        "display_height": rec.meta["display_height"],
        "fps": rec.meta["fps"],
        "rotation": rec.meta["rotation"],
        "sensor_status": "no motion data" if rec.imu is None else "loaded",
        "sensor_samples": 0 if rec.imu is None else len(rec.imu),
        "sensor_rate_hz": sensor_rate,
        "sampled_frames": None if table is None else len(table),
        "per_frame_table": None if table_path is None else str(table_path),
        "flow_valid_pairs": None if not (args.flow or args.verify) else int((table["flow_status"] == "ok").sum()),
        "gyro_alignment": None if not args.verify else table.iloc[0]["gyro_alignment"],
        "verification": verification,
        "gyro_diagnostics": diagnostic_paths,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0
