"""Whole-clip alignment diagnostics and explicit train-calibrated profile loading."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from truetake.verify_gyro import measure_alignment, VIDEO_AXES, GYRO_AXES, MAX_LAG_S


MIN_AXIS_PEAK = 0.5
MIN_WINDOW_GYRO_STD = 0.02
WINDOW_SECONDS = 5.0
WINDOW_STEP_SECONDS = 1.0
OFFSET_AGREEMENT_S = 0.2
PROFILE_VERSION = 1


def load_profile(path: str) -> dict:
    """Accept provisional mappings or thresholds with explicit train provenance."""
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("Gyro profile must be a JSON file.")
    profile = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != PROFILE_VERSION:
        raise ValueError("Gyro profile must use schema version 1.")
    threshold = profile.get("min_match_score")
    if profile.get("status") == "provisional":
        if threshold is not None:
            raise ValueError("Provisional profiles cannot set an acceptance threshold.")
    elif profile.get("status") == "calibrated":
        if (type(threshold) not in (int, float) or not np.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError("Calibrated min_match_score must be a finite number in [0,1].")
        clips = profile.get("train_clip_ids")
        if (not isinstance(clips, list) or not clips or
                any(not isinstance(clip, str) or not clip.strip() for clip in clips)
                or len(set(clips)) != len(clips)):
            raise ValueError("Calibrated profiles must record unique, nonempty train_clip_ids.")
    else:
        raise ValueError("Gyro profile status must be provisional or calibrated.")
    mapping = profile.get("axis_mapping")
    low, high = profile.get("review_low"), profile.get("review_high")
    if (low is not None or high is not None) and (type(low) not in (int, float) or type(high) not in (int, float)
            or not np.isfinite([low, high]).all() or not 0 <= low <= high <= 1):
        raise ValueError("Profile review bounds must satisfy 0 <= review_low <= review_high <= 1.")
    if (not isinstance(mapping, dict) or set(mapping) != set(VIDEO_AXES)
            or any(not isinstance(item, dict) or item.get("axis") not in GYRO_AXES
                   or type(item.get("sign")) is not int or item["sign"] not in (-1, 1)
                   for item in mapping.values())
            or len({item["axis"] for item in mapping.values()}) != len(GYRO_AXES)):
        raise ValueError("Profile must contain a signed permutation of all three gyro axes.")
    return profile


def _with_mapping(table: pd.DataFrame, mapping: dict | None) -> pd.DataFrame:
    result = table.copy()
    meta = deepcopy(result.iloc[0]["clip_meta"] or {})
    meta["gyro_calibration"] = {} if mapping is None else {
        "status": "provisional", "axis_mapping": mapping, "min_match_score": None}
    result.at[result.index[0], "clip_meta"] = meta
    return result


def fit_reference(table: pd.DataFrame) -> dict:
    """Use unchanged whole-clip lag curves; exclude axes with weak peaks."""
    baseline = deepcopy(measure_alignment(_with_mapping(table, None)).iloc[0]["gyro_alignment"])
    peaks = {axis: max((point["axis_scores"][axis] for point in baseline["lag_curve"]
                        if point["axis_scores"][axis] is not None), default=-1.0) for axis in VIDEO_AXES}
    axes = [axis for axis in VIDEO_AXES if peaks[axis] >= MIN_AXIS_PEAK]
    baseline["reference_axes"] = axes
    baseline["axis_peak_scores"] = peaks
    if baseline["status"] != "measured":
        return baseline
    if not axes:
        baseline.update(status="insufficient_motion", reason="no axis peak reaches MIN_AXIS_PEAK",
                        axis_mapping=None, offset_s=None)
        return baseline
    points = [point for point in baseline["lag_curve"] if all(point["axis_scores"][axis] is not None for axis in axes)]
    if not points:
        baseline.update(status="insufficient_overlap", reason="qualified axes have no common supported lag",
                        axis_mapping=None, offset_s=None)
        return baseline
    best = max(points, key=lambda point: np.mean([point["axis_scores"][axis] for axis in axes]))
    baseline["offset_s"] = best["lag_s"]
    baseline["reference_score"] = float(np.mean([best["axis_scores"][axis] for axis in axes]))
    baseline["at_search_boundary"] = bool(abs(best["lag_s"]) >= MAX_LAG_S)
    return baseline


def window_gyro_std(table: pd.DataFrame, start: float, stop: float) -> float:
    """Largest physical-axis standard deviation within the unshifted window."""
    values = table.loc[(table.t >= start) & (table.t < stop), list(GYRO_AXES)].to_numpy(dtype=float)
    deviations = [float(np.std(axis[np.isfinite(axis)])) for axis in values.T if np.isfinite(axis).sum() >= 2]
    return max(deviations, default=0.0)


def diagnose_alignment(table: pd.DataFrame, clip_id: str) -> pd.DataFrame:
    """Write fixed-mapping temporal validation and a draft profile on row 0."""
    table = table.copy()
    duration = float(table["sample_end"].iloc[-1])
    baseline = fit_reference(table)
    diagnostics = {"status": baseline["status"], "calibration_interval_s": [0.0, duration],
                   "baseline": baseline, "global_fixed_mapping": None,
                   "windows": [], "profile": None,
                   "note": "Overlapping windows from one recording are development diagnostics, not independent validation."}
    mapping = baseline["axis_mapping"]
    if mapping is not None:
        fixed = _with_mapping(table, mapping)
        free = _with_mapping(table, None)
        diagnostics["status"] = "diagnostic_only"
        diagnostics["global_fixed_mapping"] = measure_alignment(fixed).iloc[0]["gyro_alignment"]
        diagnostics["profile"] = {
            "schema_version": PROFILE_VERSION, "status": "provisional",
            "axis_mapping": mapping, "min_match_score": None,
            "source_clip_id": clip_id, "source_interval_s": [0.0, duration],
            "reference_axes": baseline["reference_axes"],
            "reference_offset_s": baseline["offset_s"],
            "source_rotation_deg": table.iloc[0]["clip_meta"].get("rotation"),
            "device_identity": "unconfirmed",
            "note": "Candidate only. Re-estimate offset for each recording; confirm device and orientation before reuse.",
        }
        for start in np.arange(0.0, duration - WINDOW_SECONDS + 1e-8, WINDOW_STEP_SECONDS):
            stop = float(start + WINDOW_SECONDS)
            measured = measure_alignment(fixed, interval=(float(start), stop)).iloc[0]["gyro_alignment"]
            explored = measure_alignment(free, interval=(float(start), stop)).iloc[0]["gyro_alignment"]
            reference = next((point["score"] for point in measured["lag_curve"]
                              if abs(point["lag_s"] - baseline["offset_s"]) < 1e-8), None)
            complete_mapping = len(explored["axis_scores"]) == len(VIDEO_AXES) and len(baseline["axis_scores"]) == len(VIDEO_AXES)
            std = window_gyro_std(table, float(start), stop)
            status = ("insufficient_motion" if std < MIN_WINDOW_GYRO_STD or measured["at_search_boundary"]
                      else measured["status"])
            diagnostics["windows"].append({
                "start_s": float(start), "end_s": stop, "status": status, "gyro_std": std,
                "reference_score": reference, "best_score": measured["score"],
                "best_offset_s": measured["offset_s"],
                "axis_offsets_s": measured["axis_offsets_s"],
                "lag_consistent": measured["lag_consistent"],
                "at_search_boundary": measured["at_search_boundary"],
                "overlap_pairs": measured["overlap_pairs"],
                "mapping_agrees": explored["axis_mapping"] == mapping if complete_mapping else None,
            })
    measured_windows = [window for window in diagnostics["windows"] if window["status"] == "measured"]
    mapping_windows = [window for window in measured_windows if window["mapping_agrees"] is not None]
    diagnostics["summary"] = {
        "total_windows": len(diagnostics["windows"]), "measured_windows": len(measured_windows),
        "insufficient_motion_windows": sum(window["status"] == "insufficient_motion" for window in diagnostics["windows"]),
        "reference_offset_s": baseline["offset_s"], "reference_axes": baseline["reference_axes"],
        "mapping_comparable_windows": len(mapping_windows),
        "mapping_agreement_windows": sum(window["mapping_agrees"] for window in mapping_windows),
        "offset_agreement_windows": sum(abs(window["best_offset_s"] - baseline["offset_s"]) <= OFFSET_AGREEMENT_S + 1e-8
                                         for window in measured_windows),
        "axis_consistent_windows": sum(window["lag_consistent"] for window in measured_windows),
    }
    if diagnostics["profile"] is not None:
        diagnostics["profile"]["diagnostics"] = {
            "calibration_axis_lags_consistent": baseline["lag_consistent"],
            "calibration_axis_offsets_s": baseline["axis_offsets_s"],
            "validation": diagnostics["summary"],
        }
    table["gyro_diagnostics"] = pd.Series([None] * len(table), index=table.index, dtype=object)
    table.at[table.index[0], "gyro_diagnostics"] = diagnostics
    return table
