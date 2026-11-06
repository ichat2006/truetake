"""Apply the lab's version-2 contract to grounded AI and detector evidence."""

import numpy as np
import pandas as pd

from truetake.models import CheckResult


FACE_CONFIDENCE = 0.8
FACE_CONTEXT_S = 0.5
DENSE_MAX_GAP_S = 0.75
TIME_TOLERANCE_S = 1e-6


def merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1] + TIME_TOLERANCE_S:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((float(start), float(end)))
    return result


def check_lab_quality(table: pd.DataFrame) -> tuple[list[CheckResult], dict]:
    contract, meta = table.iloc[0].contract, table.iloc[0].clip_meta
    duration = float(table.sample_end.iloc[-1])
    review = table.iloc[0].get("claude_review") or {}
    checks, decisions, cuts, privacy = [], [], [], []
    decision_messages, pending_checks = [], []
    rules = {rule["id"]: rule for rule in contract["process"] if rule.get("enabled", True)}
    assessed = {rule["rule_id"]: rule for rule in review.get("rules", [])}
    evidence = {item["id"]: item for item in review.get("evidence", [])}
    config = contract.get("duration", {})
    context = float(config.get("preserve_context_before_after_s", 1.0))

    def add(name, passed, reason, required=False, action="warning", spans=()):
        checks.append(CheckResult(name, passed, 1.0 if passed is True else 0.0, reason,
                                  [(a, b, reason) for a, b in spans]))
        if required:
            if passed is not True:
                decision_messages.append(f"{name}: {reason}")
            if passed is None:
                decisions.append("REVIEW")
                pending_checks.append((name, reason))
            elif passed is False:
                decisions.append("REJECT" if action == "reject" else "REVIEW")
                if action != "reject":
                    pending_checks.append((name, reason))

    for interval in review.get("intervals", []) if config.get("exclude_flagged_intervals", True) else []:
        rule = rules.get(interval["rule_id"], {})
        gap = float(rule.get("brief_gap_tolerance_s", rule.get("max_idle_s", 0)))
        start, end = interval["start_s"], interval["end_s"]
        if end - start <= gap + TIME_TOLERANCE_S:
            continue
        cited_times = sorted({evidence[key]["t"] for key in interval["evidence_ids"]})
        provenance = review.get("provenance", {})
        dense_ids = set(provenance.get("followup_frame_ids", []))
        confirmed = (interval["confirmed"] and provenance.get("api_calls", 0) >= 2
                     and len(cited_times) >= 2 and max(np.diff(cited_times)) <= DENSE_MAX_GAP_S
                     and cited_times[0] <= start + DENSE_MAX_GAP_S and cited_times[-1] >= end - DENSE_MAX_GAP_S
                     and all(evidence[key]["frame_id"] in dense_ids for key in interval["evidence_ids"]))
        if (not confirmed or interval.get("action") != "exclude_interval"
                or rule.get("on_long_gap", rule.get("on_failure")) != "exclude_interval"):
            continue
        # Preserve context inside each boundary rather than expanding the cut.
        a, b = start + context, end - context
        goal_times = [evidence[key]["t"] for key in review.get("goal", {}).get("evidence_ids", [])]
        if b > a and not any(a - context <= time <= b + context for time in goal_times):
            cuts.append((a, b))
    # Exact black / repeated-frame runs are local detector evidence. Sparse
    # semantic observations cannot establish these intervals by themselves.
    from truetake.measured import defaults, ranges
    limits = {**defaults()['measurement'], **contract.get('measurement', {})}
    idle_rule = rules.get('idle', {})
    if (config.get('exclude_flagged_intervals', True)
            and idle_rule.get('on_failure') == 'exclude_interval'):
        for column, maximum in [('brightness', limits['black_mean_max']),
                                ('frame_change', limits['freeze_change_max'])]:
            if column not in table:
                continue
            for start, end in ranges(table, table[column].to_numpy() <= maximum):
                if (end-start > idle_rule['max_idle_s'] and start > 0 and end < duration
                        and end-start > 2*context):
                    cuts.append((start+context, end-context))
                    add('process.local_unusable_time', True,
                        f'{column} stayed at or below {maximum:g}; preserved {context:g}s context at each edge',
                        spans=[(start,end)])
    cuts = merge_intervals(cuts)
    usable = []
    cursor = 0.
    for start, end in cuts:
        if start > cursor:
            usable.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        usable.append((cursor, duration))

    goal = review.get("goal", {})
    required_completion = bool(contract["goal"].get("required", True))
    if review.get("completion_required") is False and contract.get("operator_request"):
        required_completion = False
    verdict = {"yes": True, "no": False, "unsure": None}
    add("goal.folded", verdict.get(goal.get("verdict")), goal.get("reason", "goal review unavailable"),
        required_completion, contract["goal"].get("on_failure", "reject"))
    action = review.get("task_action_shown", {})
    add("goal.task_action_shown", verdict.get(action.get("verdict")),
        action.get("reason", "task-action review unavailable"), True, "reject")
    for name, rule in rules.items():
        assessment = assessed.get(name, {})
        passed = verdict.get(assessment.get("verdict"))
        reason = assessment.get("reason", "rule evidence unavailable")
        reported = [item for item in review.get("intervals", []) if item["rule_id"] == name]
        if passed is False and reported and all(item["end_s"] - item["start_s"] <= rule.get("brief_gap_tolerance_s", 0) + TIME_TOLERANCE_S for item in reported):
            passed = True
            reason += "; all reported gaps fall within the lab's allowed tolerance"
        elif passed is False and reported and rule.get("on_long_gap") == "exclude_interval":
            resolved = all(
                item["end_s"] - item["start_s"] <= rule.get("brief_gap_tolerance_s", 0) + TIME_TOLERANCE_S
                or any(a <= item["start_s"] + context + TIME_TOLERANCE_S and b >= item["end_s"] - context - TIME_TOLERANCE_S
                       for a, b in cuts)
                for item in reported)
            if resolved:
                passed = True
                reason += "; unusable intervals conservatively excluded; completion evidence retained"
        required = bool(rule.get("required", False))
        if not required:
            reason += "; optional rule, does not block acceptance"
        add(f"process.{name}", passed, reason, required, rule.get("on_failure", "human_review"))
    requirements = contract.get("requirements", {})
    formats = {"landscape": meta["display_width"] > meta["display_height"],
               "min_height_px": meta["display_height"] >= requirements.get("min_height_px", {}).get("value", 0),
               "min_fps": meta["fps"] + 0.1 >= requirements.get("min_fps", {}).get("value", 0)}
    for name, passed in formats.items():
        spec = requirements.get(name, {})
        if spec.get("enabled", False):
            add(f"requirements.{name}", bool(passed), f"{name}: {'met' if passed else 'not met'}",
                bool(spec.get("required", False)), spec.get("on_failure", "warning"))
    face_spec = requirements.get("faces", {})
    if face_spec.get("enabled", False):
        if "face_confidence" not in table:
            add("privacy.faces", None, "face detector not run", bool(face_spec.get("reject_whole_clip")), "reject")
        else:
            for _, row in table[table.face_confidence >= FACE_CONFIDENCE].iterrows():
                privacy.append((max(0., float(row.t) - FACE_CONTEXT_S), min(duration, float(row.t) + 0.2 + FACE_CONTEXT_S)))
            privacy = merge_intervals(privacy)
            add("privacy.faces", not bool(privacy), "detected-face intervals require blurred presentation copies" if privacy else "no faces detected in sampled frames",
                bool(face_spec.get("reject_whole_clip", False)), "reject", privacy)
    seconds = sum(b - a for a, b in usable)
    if cuts:
        add("process.excluded_time", True, "conservative exclusion supported by targeted or local unusable-frame evidence", spans=cuts)
    minimum = config.get("min_usable_seconds")
    if minimum is not None:
        add("process.usable_duration", seconds + TIME_TOLERANCE_S >= minimum,
            f"{seconds:.1f}s estimated usable; {minimum:.1f}s explicitly required", True, "reject")
    else:
        add("process.usable_duration", None if not review else True,
            "no lab minimum; usable duration is an estimate from sampled evidence")
    if not review or review.get("follow_up_requests"):
        decisions.append("REVIEW")
        decision_messages.append("AI review unavailable" if not review else "Important evidence still needs a targeted review")
    if requirements.get("sensors", {}).get("required") and not table.sensor_present.any():
        decisions.append("REVIEW")
        decision_messages.append("The lab requires motion data; none was supplied")
        pending_checks.append(("requirements.sensors", "The lab requires motion data; none was supplied"))
    status = "REJECT" if "REJECT" in decisions else "REVIEW" if "REVIEW" in decisions else "ACCEPT"
    review_items = []
    if status == "REVIEW":
        for name, reason in pending_checks:
            assessment = goal if name == "goal.folded" else action if name == "goal.task_action_shown" else assessed.get(name.removeprefix("process."), {})
            cited = [evidence[key] for key in assessment.get("evidence_ids", []) if key in evidence]
            start = max(0., min(item["t"] for item in cited) - context) if cited else 0.
            end = min(duration, max(item["t"] for item in cited) + context) if cited else duration
            review_items.append({"check": name, "reason": reason, "start_s": start, "end_s": end,
                                 "timing_kind": "evidence context" if cited else "whole clip; no localized evidence",
                                 "frame_ids": [item["frame_id"] for item in cited], "action": "inspect; retain footage pending decision"})
        for request in review.get("follow_up_requests", []):
            review_items.append({"check": "ai.follow_up", "reason": request.get("question", "Additional evidence required"),
                                 "start_s": request["start_s"], "end_s": request["end_s"],
                                 "timing_kind": "requested inspection window", "frame_ids": [],
                                 "action": "inspect; retain footage pending decision"})
    # A visible task is only a candidate until every required rule passes.
    candidate_known = action.get("verdict") in ("yes", "no")
    candidate = usable if action.get("verdict") == "yes" else []
    candidate_seconds = sum(b - a for a, b in candidate) if candidate_known else None
    approved = candidate if status == "ACCEPT" else []
    approved_seconds = sum(b - a for a, b in approved)
    reasons = decision_messages or ["Required goal and task-visibility rules met; optional warnings do not block acceptance"]
    known = [float(check.passed) for check in checks if check.passed is not None and check.name.startswith(("goal.", "process."))
             and (check.name != "goal.folded" or required_completion)
             and (not check.name.startswith("process.") or rules.get(check.name.split(".", 1)[1], {}).get("required", False))]
    return checks, {"status": status, "decision_reasons": reasons, "task_interval": (0., duration) if candidate else None,
                    "submitted_seconds": duration, "candidate_task_intervals": candidate,
                    "candidate_task_seconds": candidate_seconds, "approved_usable_seconds": approved_seconds,
                    "excluded_seconds": sum(b - a for a, b in cuts), "review_items": review_items,
                    "usable_intervals": approved, "usable_seconds": approved_seconds,
                    "usable_evidence_kind": "AI estimate from sampled evidence; not continuous ground truth",
                    "excluded_intervals": cuts, "privacy_blur_intervals": privacy,
                    "quality_score": round(100 * float(np.mean(known)), 1) if known else 0.,
                    "quality_score_kind": "required-rule agreement; not an acceptance probability",
                    "score_components": {}, "component_coverage": len(known) / max(sum(bool(rule.get('required')) for rule in rules.values()) + 1 + int(required_completion), 1),
                    "completion_required": required_completion, "visibility": {},
                    "operator_request": contract.get("operator_request", "")}
