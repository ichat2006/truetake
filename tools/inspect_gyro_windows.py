"""Print fixed-profile whole-clip and five-second window scores without changing decisions."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from truetake.calibration import load_profile, window_gyro_std, MIN_WINDOW_GYRO_STD, WINDOW_SECONDS, WINDOW_STEP_SECONDS
from truetake.detect_flow import detect_flow
from truetake.ingest import load_recording
from truetake.verify_gyro import measure_alignment, check_verification


CPU_THREADS = 2


def inspect(video: str, table_path: str, profile_path: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frame_cache = output / "frames.json"
    if frame_cache.is_file():
        table = pd.DataFrame(json.loads(frame_cache.read_text()))
    else:
        table = pd.DataFrame(json.loads(Path(table_path).read_text()))
        cv2.setNumThreads(CPU_THREADS)
        rec = load_recording(video)
        table = detect_flow(rec, table)
        table.to_json(frame_cache, orient="records", double_precision=15)
    profile = load_profile(profile_path)
    meta = dict(table.iloc[0]["clip_meta"] or {})
    meta["gyro_calibration"] = profile
    table.at[table.index[0], "clip_meta"] = meta
    measured = measure_alignment(table)
    whole = measured.iloc[0]["gyro_alignment"]
    check = asdict(check_verification(measured)[0])
    rows = []
    for start in np.arange(0., float(table.sample_end.iloc[-1]) - WINDOW_SECONDS + 1e-8, WINDOW_STEP_SECONDS):
        end = float(start + WINDOW_SECONDS)
        result_table = measure_alignment(table, interval=(float(start), end))
        result = result_table.iloc[0]["gyro_alignment"]
        status = "insufficient_motion" if window_gyro_std(table, float(start), end) < MIN_WINDOW_GYRO_STD or result["at_search_boundary"] else result["status"]
        verdict = check_verification(result_table)[0]
        rows.append({"start_s": float(start), "end_s": end, "score": result["score"],
                     "offset_s": result["offset_s"], "status": status,
                     "passed": verdict.passed if status == "measured" else None,
                     "lag_consistent": result["lag_consistent"], "boundary": result["at_search_boundary"]})
    summary = {"alignment": whole, "verification": check,
               "profile": profile_path, "measurement": "Recomputed from cached sensor table and video using unchanged verification logic",
               "note": "Window scores are diagnostic only; they do not change the whole-clip verdict."}
    (output / "whole_clip.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output / "windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["start_s", "end_s", "score", "offset_s", "status", "passed", "lag_consistent", "boundary"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"score": whole["score"], "offset_s": whole["offset_s"], "verification": check}, indent=2))
    print("start_s,end_s,score,offset_s,status,passed,lag_consistent,boundary")
    for row in rows:
        print(",".join(str(value) for value in row.values()))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--table", required=True)
    parser.add_argument("--gyro-profile", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splice_gyro"))
    args = parser.parse_args()
    inspect(args.video, args.table, args.gyro_profile, args.output_dir)
