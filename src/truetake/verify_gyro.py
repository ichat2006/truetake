"""Measure gyro/video alignment from the shared table, then check stored results."""

from itertools import permutations

import numpy as np
import pandas as pd

from truetake.models import CheckResult


LAG_STEP_S = 0.1
MAX_LAG_S = 2.0
SMOOTHING_S = 0.2
MAX_INTERPOLATION_GAP_S = 0.15
MIN_OVERLAP_S = 5.0
MIN_VALID_PAIRS = 50
MIN_GYRO_STD = 0.02
MIN_VIDEO_STD = 0.000001
MIN_INFORMATIVE_AXES = 2
MAX_AXIS_LAG_SPREAD_S = 0.2
OFFSET_WARNING_S = 1.0
MIN_MATCH_SCORE = None  # Set only after calibration on training recordings.
VIDEO_AXES = ("flow_pan", "flow_tilt", "flow_roll")
GYRO_AXES = ("gyro_gx", "gyro_gy", "gyro_gz")
WINDOW_MIN_PAIRS = 40
WINDOW_MIN_SPAN_S = 3.9
INTERPOLATION_TOLERANCE_S = 1e-8


def _interpolate(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Keep invalid samples and long gaps invalid, including at shifted lags."""
    result = np.full(len(targets), np.nan)
    finite_time = np.isfinite(times)
    times, values = times[finite_time], values[finite_time]
    if not len(times):
        return result
    right = np.minimum(np.searchsorted(times, targets), len(times) - 1)
    left = np.maximum(right - 1, 0)
    valid = ((targets >= times[0]) & (targets <= times[-1])
             & np.isfinite(values[left]) & np.isfinite(values[right])
             & (times[right] - times[left] <= MAX_INTERPOLATION_GAP_S))
    result[valid] = np.interp(targets[valid], times, values)
    # Floating-point grid roundoff must not blend an exact sample with a gap.
    exact_left = np.isclose(times[left], targets, rtol=0, atol=INTERPOLATION_TOLERANCE_S)
    exact_right = np.isclose(times[right], targets, rtol=0, atol=INTERPOLATION_TOLERANCE_S)
    result[exact_left] = values[left[exact_left]]
    result[exact_right] = values[right[exact_right]]
    return result


def prepare_signals(table: pd.DataFrame, interval: tuple[float, float] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the same smoothed video signals used for correlation and plotting."""
    flow_times = table["flow_t"].to_numpy(dtype=float)
    valid_times = flow_times[np.isfinite(flow_times)]
    if not len(valid_times):
        return np.array([]), [np.array([]) for _ in VIDEO_AXES]
    grid = np.arange(valid_times[0], valid_times[-1] + LAG_STEP_S / 2, LAG_STEP_S)
    videos = []
    for axis in VIDEO_AXES:
        values = table[axis].where(table["flow_status"] == "ok").to_numpy(dtype=float)
        if interval is not None:
            values = values.copy()
            values[(flow_times < interval[0]) | (flow_times >= interval[1])] = np.nan
        values = _interpolate(flow_times, values, grid)
        if interval is not None:
            values[(grid < interval[0]) | (grid >= interval[1])] = np.nan
        signal = pd.Series(values, index=pd.to_timedelta(grid, unit="s"))
        smooth = signal.rolling(pd.Timedelta(seconds=SMOOTHING_S), center=True,
                                closed="both").mean().to_numpy(copy=True)
        smooth[~np.isfinite(values)] = np.nan
        videos.append(smooth)
    return grid, videos


def measure_alignment(table: pd.DataFrame, interval: tuple[float, float] | None = None) -> pd.DataFrame:
    """Store diagnostic metrics on row 0. Positive lag compares v(t) with g(t+lag)."""
    table = table.copy()
    table["gyro_alignment"] = pd.Series([None] * len(table), index=table.index, dtype=object)
    result = {"status": "no_motion_data", "reason": "no motion data",
              "score": None, "offset_s": None, "axis_mapping": None,
              "axis_scores": {}, "axis_offsets_s": {}, "overlap_pairs": 0,
              "mapping_calibrated": False, "lag_consistent": False,
              "at_search_boundary": False, "lag_curve": [], "close_call": False}
    if not table["sensor_present"].any():
        table.at[table.index[0], "gyro_alignment"] = result
        return table
    calibration = (table.iloc[0]["clip_meta"] or {}).get("gyro_calibration", {})
    mapping = calibration.get("axis_mapping")
    if mapping is not None:
        if (set(mapping) != set(VIDEO_AXES)
                or {item["axis"] for item in mapping.values()} != set(GYRO_AXES)
                or any(item["sign"] not in (-1, 1) for item in mapping.values())):
            raise ValueError("Calibration requires a signed permutation of the three gyro axes.")
        result["mapping_calibrated"] = calibration.get("status") == "calibrated"
    grid, videos = prepare_signals(table, interval)
    min_pairs, min_span = MIN_VALID_PAIRS, MIN_OVERLAP_S
    if interval is not None:
        start, end = interval
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError("Alignment interval must have finite bounds with 0 <= start < end.")
        selected = (grid >= start) & (grid < end)
        grid, videos = grid[selected], [video[selected] for video in videos]
        min_pairs, min_span = WINDOW_MIN_PAIRS, WINDOW_MIN_SPAN_S
    result.update(status="insufficient_overlap", reason="not enough valid motion overlap")
    if len(grid) < min_pairs:
        table.at[table.index[0], "gyro_alignment"] = result
        return table
    lags = np.arange(-round(MAX_LAG_S / LAG_STEP_S), round(MAX_LAG_S / LAG_STEP_S) + 1) * LAG_STEP_S
    correlations = np.full((len(lags), 3, 3), np.nan)
    counts = np.zeros((len(lags), 3, 3), dtype=int)
    for lag_index, lag in enumerate(lags):
        for gyro_index, gyro_axis in enumerate(GYRO_AXES):
            gyro = _interpolate(table["t"].to_numpy(), table[gyro_axis].to_numpy(), grid + lag)
            for video_index, video in enumerate(videos):
                valid = np.isfinite(video) & np.isfinite(gyro)
                if valid.sum() < min_pairs or np.ptp(grid[valid]) < min_span - 1e-8:
                    continue
                v, g = video[valid], gyro[valid]
                if np.std(v) < MIN_VIDEO_STD or np.std(g) < MIN_GYRO_STD:
                    continue
                correlations[lag_index, video_index, gyro_index] = np.corrcoef(v, g)[0, 1]
                counts[lag_index, video_index, gyro_index] = valid.sum()
    candidates = [tuple(GYRO_AXES.index(mapping[axis]["axis"]) for axis in VIDEO_AXES)] if mapping else permutations(range(3))
    best = None
    for assignment in candidates:
        for lag_index in range(len(lags)):
            scores = correlations[lag_index, np.arange(3), assignment]
            signs = np.array([mapping[axis]["sign"] for axis in VIDEO_AXES]) if mapping else np.where(scores < 0, -1, 1)
            signed = scores * signs
            valid = np.isfinite(signed)
            if valid.sum() < MIN_INFORMATIVE_AXES:
                continue
            score = float(signed[valid].mean())
            if best is None or score > best[0]:
                best = (score, lag_index, assignment, signs, valid)
    if best is None:
        result.update(status="insufficient_motion", reason="not enough motion to verify, or insufficient valid overlap")
    else:
        score, lag_index, assignment, signs, valid = best
        result.update(status="measured", reason="alignment measured; calibration required for a verdict",
                      score=score, offset_s=float(lags[lag_index]), axis_mapping={},
                      at_search_boundary=bool(lag_index in (0, len(lags) - 1)))
        overlaps = []
        for index in range(3):
            axis = VIDEO_AXES[index]
            result["axis_mapping"][axis] = {"axis": GYRO_AXES[assignment[index]], "sign": int(signs[index])}
            if not valid[index]:
                continue
            curve = correlations[:, index, assignment[index]] * signs[index]
            peak = int(np.nanargmax(curve))
            result["axis_scores"][axis] = float(curve[lag_index])
            result["axis_offsets_s"][axis] = float(lags[peak])
            overlaps.append(int(counts[lag_index, index, assignment[index]]))
        # Only the two strongest measured axes decide consistency. A stabilized
        # weak axis must not veto the camera's other two independent signals.
        strongest = sorted(result["axis_scores"], key=result["axis_scores"].get, reverse=True)[:MIN_INFORMATIVE_AXES]
        offsets = [result["axis_offsets_s"][axis] for axis in strongest]
        result["lag_consistent"] = bool(len(offsets) == MIN_INFORMATIVE_AXES
                                        and np.ptp(offsets) <= MAX_AXIS_LAG_SPREAD_S + 1e-8)
        result["overlap_pairs"] = min(overlaps)
        for index, lag in enumerate(lags):
            scores = correlations[index, np.arange(3), assignment] * signs
            finite = np.isfinite(scores)
            result["lag_curve"].append({
                "lag_s": float(lag),
                "score": float(scores[finite].mean()) if finite.sum() >= MIN_INFORMATIVE_AXES else None,
                "axis_scores": {axis: float(value) if np.isfinite(value) else None
                                for axis, value in zip(VIDEO_AXES, scores)},
            })
    low, high = calibration.get("review_low"), calibration.get("review_high")
    result["close_call"] = bool(result["status"] == "measured" and low is not None and high is not None
                                and low <= result["score"] <= high)
    table.at[table.index[0], "gyro_alignment"] = result
    return table


def check_verification(table: pd.DataFrame) -> list[CheckResult]:
    """Read stored table evidence only; an exploratory mapping never earns a pass."""
    result = table.iloc[0]["gyro_alignment"]
    threshold = (table.iloc[0]["clip_meta"] or {}).get("gyro_calibration", {}).get("min_match_score", MIN_MATCH_SCORE)
    passed = None
    reason = result["reason"]
    flags = []
    if result["status"] == "measured":
        if abs(result["offset_s"]) > OFFSET_WARNING_S:
            flags.append((0.0, float(table["sample_end"].iloc[-1]), "clock offset exceeds 1 second"))
        if not result["mapping_calibrated"] or threshold is None:
            reason = "axis mapping or match threshold is uncalibrated"
        elif not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Calibrated match threshold must be between zero and one.")
        elif result["at_search_boundary"]:
            reason = "best offset is at the search boundary"
        elif result["overlap_pairs"] < MIN_VALID_PAIRS:
            reason = "not enough valid motion overlap"
        elif result["score"] < threshold:
            passed = False
            reason = "phone motion doesn't match the video"
        elif not result["lag_consistent"]:
            reason = "video axes disagree on clock offset"
        else:
            passed = bool(result["score"] >= threshold)
            reason = "gyro agrees with video" if passed else "gyro does not match video"
    return [CheckResult("verification.gyro", passed, result["score"] if result["score"] is not None else 0.0, reason, flags)]
