"""Contract checks consume table evidence; intervals use float clip seconds."""

import numpy as np
import pandas as pd

from truetake.models import CheckResult


MIN_VISIBILITY_FRACTION = 0.9
MAX_PEOPLE_HOLD_S = 0.21
MAX_VISIBILITY_GAP_S = 0.5
MIN_ACTIVE_RUN_S = 1.0
MIN_HAND_SPEED = 0.02
IDLE_CHANGE = 0.01
IDLE_GYRO_RMS = 0.02
MAX_IDLE_S = 3.0
FACE_THRESHOLD = 0.8
CLOTHING_MARGIN = 0.02
MAX_CLIP_HOLD_S = 0.51
FPS_TOLERANCE = 0.1
BRIGHTNESS_REFERENCE = 100.0
SHARPNESS_REFERENCE = 100.0
INTERVAL_TOLERANCE = 1e-6


def intervals(table: pd.DataFrame, mask: np.ndarray) -> list[tuple[float, float]]:
    spans = []
    for row in np.flatnonzero(mask):
        start = float(table.t.iloc[row])
        end = min(float(table.sample_end.iloc[row]), start + 0.1)
        if spans and start <= spans[-1][1] + INTERVAL_TOLERANCE:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def held_values(table: pd.DataFrame, column: str, status: str, hold_s: float) -> np.ndarray:
    values = np.full(len(table), np.nan)
    if column not in table or status not in table:
        return values
    scheduled = table[table[status] != "not_sampled"]
    if scheduled.empty:
        return values
    t = table.t.to_numpy()
    indices = np.searchsorted(scheduled.t.to_numpy(), t, side="right") - 1
    valid = indices >= 0
    selected = np.maximum(indices, 0)
    valid &= (t - scheduled.t.to_numpy()[selected] <= hold_s)
    valid &= scheduled[status].to_numpy()[selected] == "ok"
    values[valid] = scheduled[column].to_numpy(dtype=float)[selected[valid]]
    return values


def check_quality(table: pd.DataFrame) -> tuple[list[CheckResult], dict]:
    if (table.iloc[0].contract or {}).get("schema_version", 1) == 2:
        from truetake.lab_quality import check_lab_quality
        return check_lab_quality(table)
    meta, contract = table.iloc[0].clip_meta, table.iloc[0].contract or {}
    duration = float(table.sample_end.iloc[-1])
    checks = []

    def add(name, passed, score, reason, spans=()):
        checks.append(CheckResult(name, passed, float(score), reason,
                                  [(start, end, reason) for start, end in spans]))

    hard = contract.get("requirements", {}).get("hard", [])
    format_results = {
        "landscape": meta["display_width"] > meta["display_height"],
        "min_1080p": min(meta["display_width"], meta["display_height"]) >= 1080 and max(meta["display_width"], meta["display_height"]) >= 1920,
        "min_30fps": meta["fps"] + FPS_TOLERANCE >= 30,
    }
    for name, passed in format_results.items():
        if name in hard:
            add(f"requirements.{name}", bool(passed), float(passed),
                f"{name}: {'met' if passed else 'failed'}", [] if passed else [(0., duration)])
    faces = table.face_confidence.dropna() if "face_confidence" in table else pd.Series(dtype=float)
    if "no_faces" in hard:
        face_spans = intervals(table, table.face_confidence.fillna(0).to_numpy() >= FACE_THRESHOLD) if len(faces) else []
        add("requirements.no_faces", not bool(face_spans) if len(faces) else None,
            1 - float(bool(face_spans)) if len(faces) else 0, "face detected" if face_spans else "no face detected in sampled frames" if len(faces) else "face detector not run", face_spans)
    for name in hard:
        if name not in {*format_results, "no_faces"}:
            reason = "no motion data" if name == "not_on_tripod" and not table.sensor_present.any() else "not established by available evidence"
            add(f"requirements.{name}", None, 0, reason)
    hands = held_values(table, "hands_inside", "people_status", MAX_PEOPLE_HOLD_S)
    counts = held_values(table, "hand_count", "people_status", MAX_PEOPLE_HOLD_S)
    motion = held_values(table, "hand_motion", "people_status", MAX_PEOPLE_HOLD_S)
    active = (counts > 0) & (motion >= MIN_HAND_SPEED)
    runs = [(a, b) for a, b in intervals(table, active) if b - a >= MIN_ACTIVE_RUN_S - INTERVAL_TOLERANCE]
    task = (runs[0][0], runs[-1][1]) if runs else None
    usable = []
    estimated_usable = []
    garment_calibrated = bool(meta.get("garment_threshold_calibrated", False))
    fractions = {"hands_visible": None, "garment_visible": None}
    activity_score = None
    if task is None:
        add("process.task_window", None, 0, "no sustained hand activity identified")
    else:
        inside = (table.t.to_numpy() >= task[0]) & (table.t.to_numpy() < task[1] - INTERVAL_TOLERANCE)
        for a, b in intervals(table, hands == 0):
            previous = table.t.to_numpy() < a
            following = table.t.to_numpy() >= b - INTERVAL_TOLERANCE
            if b - a < MAX_VISIBILITY_GAP_S and previous.any() and following.any():
                if hands[np.flatnonzero(previous)[-1]] == 1 and hands[np.flatnonzero(following)[0]] == 1:
                    hands[(table.t >= a) & (table.t < b - INTERVAL_TOLERANCE)] = 1
        clothing = held_values(table, "clothing_similarity", "clip_status", MAX_CLIP_HOLD_S)
        empty = held_values(table, "no_clothing_similarity", "clip_status", MAX_CLIP_HOLD_S)
        garment = np.where(np.isfinite(clothing), (clothing - empty >= CLOTHING_MARGIN).astype(float), np.nan)
        good = inside.copy()
        for name, signal in (("hands_visible", hands), ("garment_visible", garment)):
            known = inside & np.isfinite(signal)
            fraction = float(np.mean(signal[inside])) if np.isfinite(signal[inside]).all() else None
            calibrated = name != "garment_visible" or garment_calibrated
            fractions[name] = fraction if calibrated else None
            passed = None if fraction is None or not calibrated else fraction >= MIN_VISIBILITY_FRACTION
            add(f"process.{name}", passed, fraction or 0,
                f"{name}: uncalibrated CLIP estimate; human review required" if not calibrated else
                f"{name}: {fraction:.0%} of task samples" if fraction is not None else f"{name}: incomplete evidence",
                intervals(table, inside & (signal == 0)) if calibrated else [])
            good &= known & (signal == 1)
        idle = (table.frame_change.to_numpy() <= IDLE_CHANGE) & (motion < MIN_HAND_SPEED)
        if table.sensor_present.any():
            gyro = np.sqrt((table[["gx", "gy", "gz"]].to_numpy() ** 2).mean(axis=1))
            idle &= gyro < IDLE_GYRO_RMS
        idle_spans = [(a, b) for a, b in intervals(table, inside & idle) if b - a > MAX_IDLE_S]
        for a, b in idle_spans:
            good &= ~((table.t >= a) & (table.t < b - INTERVAL_TOLERANCE)).to_numpy()
        activity_score = 1 - sum(b - a for a, b in idle_spans) / (task[1] - task[0])
        add("process.continuous", not bool(idle_spans), activity_score,
            "idle stretch over 3 seconds" if idle_spans else "no measured idle stretch over 3 seconds", idle_spans)
        estimated_usable = intervals(table, good)
        usable = estimated_usable if garment_calibrated else []
    usable_seconds = sum(b - a for a, b in usable)
    minimum = float(contract.get("min_usable_seconds", 10))
    coverage_complete = task is not None and all(value is not None for value in fractions.values())
    enough = usable_seconds + INTERVAL_TOLERANCE >= minimum
    add("process.usable_duration", enough if coverage_complete else None, min(1, usable_seconds / max(minimum, INTERVAL_TOLERANCE)),
        f"{usable_seconds:.1f}s confirmed usable; {minimum:.1f}s required")
    review = table.iloc[0].get("claude_review") or {}
    goal = review.get("goal", {})
    verdict = goal.get("verdict", "unsure")
    add("goal.folded", {"yes": True, "no": False, "unsure": None}[verdict], float(verdict == "yes"),
        goal.get("reason", "Claude review pending"))
    components = {"lighting": float(np.clip(table.brightness.mean() / BRIGHTNESS_REFERENCE, 0, 1)),
                  "sharpness": float(np.clip(table.blur_var.mean() / SHARPNESS_REFERENCE, 0, 1)),
                  "hands": fractions["hands_visible"], "activity": activity_score}
    observed = [value for value in components.values() if value is not None]
    hard_fail = any(check.passed is False and check.name.startswith("requirements.") for check in checks)
    rejected = hard_fail or verdict == "no" or duration < minimum or (coverage_complete and not enough)
    unresolved_hard = any(check.passed is None and check.name.startswith("requirements.") for check in checks)
    status = "REJECT" if rejected else "ACCEPT" if verdict == "yes" and coverage_complete and enough and not unresolved_hard else "REVIEW"
    return checks, {"status": status, "task_interval": task, "usable_intervals": usable,
                    "usable_seconds": usable_seconds, "quality_score": round(100 * float(np.mean(observed)), 1),
                    "quality_score_kind": "provisional observed-quality proxy; not calibrated acceptance confidence",
                    "estimated_usable_seconds": sum(b - a for a, b in estimated_usable),
                    "score_components": components, "component_coverage": len(observed) / len(components),
                    "visibility": fractions}
