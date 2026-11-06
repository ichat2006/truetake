"""Train-only CPU gyro/video swap experiment. No Claude calls or test media reads."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from truetake.calibration import load_profile
from truetake.detect_basic import detect_basic
from truetake.detect_flow import detect_flow
from truetake.frame_table import build_table
from truetake.ingest import load_recording
from truetake.verify_gyro import measure_alignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/gyro_swap_eval.csv"
HISTOGRAM_BINS = 20
PLOT_DPI = 120
MIN_PARTICIPANTS = 2
CSV_FIELDS = ["clip_id", "pairing", "score", "offset_s", "status"]


def train_entries(manifest: Path) -> list[dict]:
    """Filter before resolving/statting/opening ANY referenced media or sensors."""
    with manifest.open(newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if row["split"] == "train" and row["sensors_path"]
                and row["attack"] == "none"]


def with_profile(table: pd.DataFrame, profile: dict) -> pd.DataFrame:
    table = table.copy()
    meta = dict(table.iloc[0].clip_meta or {})
    if profile.get("source_rotation_deg") not in (None, meta.get("rotation")):
        raise ValueError("Gyro profile rotation differs from the recording")
    meta["gyro_calibration"] = profile
    table.at[table.index[0], "clip_meta"] = meta
    return table


def score_pair(real: pd.DataFrame, swapped: pd.DataFrame, profile: dict | None = None) -> list[dict]:
    """Lock the same mapping for both pairings; never optimize a swap's signs."""
    measured = measure_alignment(with_profile(real, profile) if profile else real)
    result = measured.iloc[0].gyro_alignment
    mapping = profile or {"status": "provisional", "min_match_score": None, "axis_mapping": result["axis_mapping"]}
    if mapping["axis_mapping"] is None:
        return [{"pairing": pairing, "score": None, "offset_s": None, "status": result["status"]}
                for pairing in ("real", "swapped")]
    other = measure_alignment(with_profile(swapped, mapping)).iloc[0].gyro_alignment
    return [{"pairing": pairing, **{key: value[key] for key in ("score", "offset_s", "status")}}
            for pairing, value in (("real", result), ("swapped", other))]


def suggest_threshold(rows: list[dict]) -> dict:
    """Prefer the midpoint of a separation gap; report overlap instead of hiding it."""
    groups = {kind: [row["score"] for row in rows if row["pairing"] == kind
                     and row["status"] == "measured" and row["score"] is not None
                     and np.isfinite(row["score"])] for kind in ("real", "swapped")}
    if not all(groups.values()):
        return {"suggested_threshold": None, "separated": False, "reason": "need measured real and swapped groups"}
    real, swapped = groups["real"], groups["swapped"]
    separated = min(real) > max(swapped)
    midpoint = (min(real) + max(swapped)) / 2 if separated else (float(np.median(real)) + float(np.median(swapped))) / 2
    usable = separated and 0 <= midpoint <= 1
    return {"suggested_threshold": float(midpoint), "separated": separated, "usable_separation": usable,
            "real_range": [float(min(real)), float(max(real))],
            "swapped_range": [float(min(swapped)), float(max(swapped))],
            "method": "midpoint between closest group edges" if separated else "midpoint of medians; groups overlap",
            "note": "Training development suggestion only; not an automatically calibrated profile."}


def plot_scores(rows: list[dict], output: Path) -> None:
    figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bins = np.linspace(-1, 1, HISTOGRAM_BINS + 1)
    for kind in ("real", "swapped"):
        values = [row["score"] for row in rows if row["pairing"] == kind
                  and row["status"] == "measured" and row["score"] is not None]
        axis.hist(values, bins=bins, alpha=0.55, label=f"{kind} (n={len(values)})")
    threshold = suggest_threshold(rows)["suggested_threshold"]
    if threshold is not None:
        axis.axvline(threshold, color="black", linestyle="--", label=f"suggestion {threshold:.3f}")
    axis.set(xlabel="Whole-clip match score (correlation)", ylabel="Pairings",
             title="Train only: own vs other-participant sensors")
    axis.legend()
    figure.savefig(output, dpi=PLOT_DPI)


def flow_table(rec, sensor_path: Path, cache_dir: Path) -> pd.DataFrame:
    checksum = hashlib.sha256()
    paths = [Path(rec.video_path), sensor_path] + [ROOT / "src/truetake" / name for name in
             ("detect_flow.py", "detect_basic.py", "frame_table.py", "ingest.py")]
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (checksum.hexdigest() + ".json")
    if path.is_file():
        return pd.DataFrame(json.loads(path.read_text()))
    table = detect_flow(rec, detect_basic(rec, build_table(rec)))
    path.write_text(table.to_json(orient="records", double_precision=15))
    return table


def run_evaluation(manifest: Path, output: Path, profile_path: str | None = None) -> dict:
    if output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Evaluation outputs must be outside data/")
    entries = train_entries(manifest)
    profile = load_profile(profile_path) if profile_path else None
    if profile and not set(profile.get("train_clip_ids", [])) <= {entry["clip_id"] for entry in entries}:
        raise ValueError("Profile train_clip_ids must refer only to sensor-equipped train entries")
    available, skipped = [], []
    for row in entries:
        if "unknown" in row["participant"].lower() or not row["participant"].strip():
            skipped.append({"clip_id": row["clip_id"], "reason": "participant not established"})
            continue
        if not (ROOT / row["video_path"]).is_file() or not (ROOT / row["sensors_path"]).is_file():
            skipped.append({"clip_id": row["clip_id"], "reason": "train media/sensors not downloaded"})
            continue
        available.append(row)
    if len({row["participant"] for row in available}) < MIN_PARTICIPANTS:
        raise ValueError("Need downloaded sensor-equipped train clips from at least two known participants")
    output.parent.mkdir(parents=True, exist_ok=True)
    recordings, tables = {}, {}
    for row in available:
        clip_id = row["clip_id"]
        rec = load_recording(str(ROOT / row["video_path"]), str(ROOT / row["sensors_path"]))
        recordings[clip_id] = rec
        tables[clip_id] = flow_table(rec, ROOT / row["sensors_path"], output.parent / "gyro_swap_eval_cache")
    results, pairings = [], []
    for row in available:
        rec = recordings[row["clip_id"]]
        candidates = [other for other in available if other["participant"] != row["participant"]]
        other = min(candidates, key=lambda other: (abs(recordings[other["clip_id"]].meta["duration_s"] - rec.meta["duration_s"]), other["clip_id"]))
        # Rebuild only sensor columns using the original video's timing, then
        # reuse that video's flow. No sensor time scaling, extrapolation or test reads.
        swapped = build_table(replace(rec, imu=recordings[other["clip_id"]].imu))
        for column in tables[row["clip_id"]]:
            if column.startswith("flow_"):
                swapped[column] = tables[row["clip_id"]][column].to_numpy()
        for score in score_pair(tables[row["clip_id"]], swapped, profile):
            results.append({"clip_id": row["clip_id"], **score})
        pairings.append({"clip_id": row["clip_id"], "swapped_sensor_clip_id": other["clip_id"],
                         "participant": row["participant"], "other_participant": other["participant"]})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    plot_scores(results, output.with_suffix(".png"))
    summary = {**suggest_threshold(results), "train_clip_ids": [row["clip_id"] for row in available],
               "pairings": pairings, "skipped_train_entries": skipped,
               "mapping_policy": "provided profile" if profile else "fit on own train clip; locked for its swap; development only",
               "csv": str(output), "histogram": str(output.with_suffix(".png"))}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gyro-profile", help="Optional fixed mapping; locks the same profile for both groups")
    args = parser.parse_args(argv)
    try:
        run_evaluation(args.manifest, args.output, args.gyro_profile)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"gyro swap evaluation: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
