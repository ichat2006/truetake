"""Final task decisions from measured bands, grounded goal evidence and gyro checks."""
from truetake.measured import context, defaults
from truetake.models import CheckResult

STATUS = {"APPROVED": "ACCEPT", "NEEDS REVIEW": "REVIEW", "REJECTED": "REJECT"}

def apply_verdict(table, checks: list[CheckResult], quality: dict):
    contract = table.iloc[0].get("contract") or {}
    if contract.get("schema_version") != 2 or not contract.get("goal"):
        return checks, quality
    measured = context(table, contract, quality.get("excluded_intervals", []))
    failures, reviews = [], []
    timestamp = 0.
    for rule in measured["rules"]:
        label = rule["rule_id"].replace("_", " ").capitalize()
        value = rule["value"]
        percent = "not measured" if value is None else f"{100*value:.0f}% of task time"
        window = rule["flags"][0] if rule["flags"] else [0., float(table.sample_end.iloc[-1])]
        reason = (f"{label}: {percent} (needs {100*rule['target']:.0f}%; "
                  f"{100*rule['lower']:.0f}–{100*rule['upper']:.0f}% is a close call) — check {window[0]:.1f}–{window[1]:.1f}s")
        state = rule["state"]
        checks.append(CheckResult("measured."+rule["rule_id"], {"pass":True,"fail":False,"review":None}[state],
                                  value or 0., reason, [(window[0],window[1],reason)]))
        if state == "fail": failures.append(reason);timestamp=window[0]
        elif state == "review": reviews.append(reason);timestamp=window[0]
    seconds = measured["usable_assessment"]
    if seconds and seconds["state"] != "pass":
        reason = (f"Usable time {seconds['value']:.1f}s (needs {seconds['target']:.1f}s; "
                  f"{seconds['lower']:.1f}–{seconds['upper']:.1f}s is a close call) — check 0:00")
        (failures if seconds["state"] == "fail" else reviews).append(reason)
    review = table.iloc[0].get("claude_review") or {}
    goal = review.get("goal", {})
    if goal.get("verdict") == "no":
        failures.append("Goal not met — " + goal.get("reason", "required outcome not shown") + " — check 0:00")
    elif goal.get("verdict") != "yes":
        reviews.append("Goal is not confirmed by valid evidence — check 0:00")
    semantic = {"process."+rule["id"] for rule in contract.get("process", []) if rule.get("required") and "measurement" not in rule}
    for check in checks:
        hard = check.name.startswith("requirements.") and contract.get("requirements", {}).get(check.name.split(".",1)[1], {}).get("required")
        if (hard or check.name in semantic or check.name == "goal.task_action_shown") and check.passed is False:
            failures.append(check.reason + " — check 0:00")
        elif (hard or check.name in semantic or check.name == "goal.task_action_shown") and check.passed is None:
            reviews.append(check.reason + " — check 0:00")
    gyro = next((check for check in checks if check.name == "verification.gyro"), None)
    profile = (table.iloc[0].get("clip_meta") or {}).get("gyro_calibration", {})
    if gyro and gyro.passed is False:
        failures.insert(0, f"Phone motion score {gyro.score:.3f} below {profile.get('min_match_score',0.):.3f} — check 0:00")
    elif gyro and measured["close_calls_enabled"] and profile.get("review_low") is not None and profile["review_low"] <= gyro.score <= profile["review_high"]:
        reviews.append(f"Phone motion score {gyro.score:.3f} is in the {profile['review_low']:.3f}–{profile['review_high']:.3f} close-call zone — check 0:00")
    if not measured["close_calls_enabled"] and reviews:
        failures += [reason for reason in reviews if reason.startswith("Goal") or "not measured" in reason]
        reviews = []
    verdict = "REJECTED" if failures else "NEEDS REVIEW" if reviews else "APPROVED"
    result = {**quality, "status": STATUS[verdict], "verdict": verdict, "measured": measured,
              "decision_reasons": failures or reviews or ["Meets the task checklist."],
              "decision_timestamp_s": timestamp, "sensor_verified": bool(gyro and gyro.passed is True)}
    usable = quality.get("candidate_task_intervals") or quality.get("usable_intervals") or []
    result["usable_intervals"] = usable if verdict == "APPROVED" else []
    result["usable_seconds"] = sum(b-a for a,b in result["usable_intervals"])
    result["approved_usable_seconds"] = result["usable_seconds"]
    return checks, result
