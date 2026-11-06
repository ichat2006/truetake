"""Choose compact, chronological task evidence without deciding the task itself."""

from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd

from truetake.detect_basic import iter_sampled_frames


SELECTOR_VERSION = 1
FOLLOWUP_POLICY_VERSION = 1
OVERVIEW_FRAMES = 8
END_CONTEXT_S = 1.0
DARK_BRIGHTNESS = 20.0
LOW_SHARPNESS = 8.0
LOW_CHANGE = 0.002
MIN_VISUAL_WINDOW_S = 1.0
MIN_IDLE_WINDOW_S = 3.0
MAX_WINDOWS = 3
MAX_SAMPLE_GAP_S = 0.21
JPEG_QUALITY = 85


def candidate_windows(table: pd.DataFrame) -> list[dict]:
    """Missing measurements break a run; warnings never authorize cuts."""
    kinds = [("dark", "brightness", DARK_BRIGHTNESS, MIN_VISUAL_WINDOW_S),
             ("low_detail", "blur_var", LOW_SHARPNESS, MIN_VISUAL_WINDOW_S),
             ("little_image_change", "frame_change", LOW_CHANGE, MIN_IDLE_WINDOW_S)]
    windows = []
    for kind, column, threshold, minimum in kinds:
        if column not in table:
            continue
        start, end = None, None
        for _, row in table.iterrows():
            valid = pd.notna(row[column]) and row[column] < threshold
            if start is not None and (not valid or row.t - end > MAX_SAMPLE_GAP_S):
                if end - start >= minimum:
                    windows.append({"kind": kind, "start_s": start, "end_s": end,
                                    "reason": "Measurement suggests inspection; not a task failure or confirmed cut"})
                start = None
            if valid:
                if start is None:
                    start = float(row.t)
                end = min(float(row.sample_end), float(row.t) + MAX_SAMPLE_GAP_S)
        if start is not None and end - start >= minimum:
            windows.append({"kind": kind, "start_s": start, "end_s": end,
                            "reason": "Measurement suggests inspection; not a task failure or confirmed cut"})
    return sorted(windows, key=lambda w: (-(w["end_s"] - w["start_s"]), w["start_s"]))[:MAX_WINDOWS]


def select_evidence(table: pd.DataFrame) -> pd.DataFrame:
    """Preserve beginning/end context and cover each temporal quarter."""
    table = table.copy()
    times = table.t.to_numpy()
    duration = float(table.sample_end.iloc[-1])
    windows = candidate_windows(table)
    chosen = {}

    def add(index, reason):
        row = int(table.index[index])
        chosen[row] = chosen.get(row, []) + [reason]

    for target, reason in ((times[0], "clip start"), (min(END_CONTEXT_S, times[-1]), "start context"),
                           (max(times[0], times[-1] - END_CONTEXT_S), "ending context"), (times[-1], "clip end")):
        add(int(np.argmin(np.abs(times - target))), reason)
    for quarter in range(4):
        left, right = duration * quarter / 4, duration * (quarter + 1) / 4
        available = [i for i, t in enumerate(times) if left <= t < right and int(table.index[i]) not in chosen]
        if not available:
            continue
        relevant = [w for w in windows if w["start_s"] < right and w["end_s"] > left]
        target = (max(left, relevant[0]["start_s"]) + min(right, relevant[0]["end_s"])) / 2 if relevant else (left + right) / 2
        index = min(available, key=lambda i: abs(times[i] - target))
        add(index, "candidate-window context" if relevant else "temporal coverage")
    for index in np.linspace(0, len(table) - 1, min(OVERVIEW_FRAMES, len(table))).round().astype(int):
        if len(chosen) >= min(OVERVIEW_FRAMES, len(table)):
            break
        if int(table.index[index]) not in chosen:
            add(int(index), "temporal coverage")
    table["claude_selected"] = table.index.isin(chosen)
    table["claude_selection_reason"] = pd.Series({row: "; ".join(reasons) for row, reasons in chosen.items()}, dtype=object)
    table["evidence_plan"] = pd.Series([None] * len(table), index=table.index, dtype=object)
    table.at[table.index[0], "evidence_plan"] = {
        "mode": "focused", "selector_version": SELECTOR_VERSION,
        "followup_policy_version": FOLLOWUP_POLICY_VERSION,
        "initial_frame_ids": sorted(chosen), "candidate_windows": windows,
        "limitations": "Sparse observations: do not infer continuous visibility or precise cuts. Hands are not a gate. Unchanged images can show a valid finished task."}
    return table


def route_followups(review: dict, contract: dict) -> dict:
    """Spend extra calls on unresolved required judgments, not optional checks."""
    review = deepcopy(review)
    requests = review.get("follow_up_requests", [])
    if not requests:
        return review
    completion = contract["goal"].get("required", True)
    if review.get("completion_required") is False and contract.get("operator_request"):
        completion = False
    goal, action = review.get("goal", {}).get("verdict"), review.get("task_action_shown", {}).get("verdict")
    rules = {item["rule_id"]: item.get("verdict") for item in review.get("rules", [])}
    required = [item["id"] for item in contract["process"] if item.get("required") and item.get("enabled", True)]
    terminal_failure = action == "no" or completion and goal == "no"
    unresolved = (action not in ("yes", "no") or completion and goal not in ("yes", "no")
                  or any(rules.get(name) != "yes" for name in required))
    if terminal_failure or not unresolved:
        review["deferred_follow_up_requests"] = requests
        review["follow_up_requests"] = []
        review["follow_up_routing_reason"] = ("Required task failure already determines rejection" if terminal_failure
                                                else "Required judgments are decided; optional uncertainty does not trigger another call")
    return review


def save_selected_frames(rec, table: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Extract only selected upright images, retaining original clip timestamps."""
    table = table.copy()
    destination = Path(output_dir) / "selected"
    destination.mkdir(parents=True, exist_ok=True)
    for row, frame in iter_sampled_frames(rec, table[table.claude_selected]):
        path = destination / f"frame_{row:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise ValueError("Could not save selected evidence image.")
        table.at[row, "thumbnail_path"] = str(path.resolve())
    return table
