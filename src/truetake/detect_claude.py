"""One cached Claude review per clip; absent credentials remain explicit unknowns."""

import base64
import hashlib
import json
import os
import math
from pathlib import Path

import anthropic
import pandas as pd
import numpy as np
import cv2
import yaml
from truetake.detect_basic import iter_sampled_frames
from truetake.evidence import route_followups


MODEL = "claude-haiku-4-5"
PROMPT_VERSION = "review_v1"
MAX_TOKENS = 4000
TIMEOUT_S = 45.0
MAX_FOLLOWUPS = 2
MAX_WINDOW_S = 8.0
FOLLOWUP_HZ = 2.0
MAX_FOLLOWUP_FRAMES = 32
STEPS = {"smooth", "sleeves-in", "fold-half-long", "fold-half-short", "fold-thirds", "roll", "flip", "drop", "recover", "other"}


def parse_review(text: str) -> dict:
    """Accept a JSON object with an optional Markdown fence or prose wrapper."""
    start = text.find("{")
    if start < 0:
        raise ValueError("Review contains no JSON object.")
    result, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(result, dict):
        raise ValueError("Review must be a JSON object.")
    for item in result.get("evidence", []):
        frame_id = item.get("frame_id")
        if isinstance(frame_id, str) and frame_id.isascii() and frame_id.isdecimal():
            item["frame_id"] = int(frame_id)
    return result


def bound_followups(review: dict) -> dict:
    """Cap requested extra image coverage; never alter observations or verdicts."""
    requests = review.get("follow_up_requests", [])
    if isinstance(requests, list):
        for item in requests[:MAX_FOLLOWUPS]:
            start, end = item.get("start_s"), item.get("end_s")
            if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                    and math.isfinite(start) and math.isfinite(end) and end - start > MAX_WINDOW_S):
                item["start_s"] = end - MAX_WINDOW_S
    return review


def validate_review(review: dict, frames: dict[int, float]) -> dict:
    if not isinstance(review, dict) or not isinstance(review.get("evidence"), list):
        raise ValueError("Claude response requires an evidence list.")
    ids = set()
    for item in review["evidence"]:
        if (type(item["frame_id"]) is not int or item["frame_id"] not in frames or not math.isfinite(item["t"])
                or abs(item["t"] - frames[item["frame_id"]]) > 0.001
                or not isinstance(item["observation"], str) or not isinstance(item["id"], str)
                or item["id"] in ids):
            raise ValueError("Claude evidence must cite a supplied frame and timestamp.")
        ids.add(item["id"])
    goal = review["goal"]
    if goal["verdict"] not in ("yes", "no", "unsure") or not isinstance(goal["reason"], str):
        raise ValueError("Invalid goal verdict.")
    if not isinstance(goal.get("evidence_ids"), list) or not set(goal["evidence_ids"]) <= ids:
        raise ValueError("Unknown goal evidence reference.")
    if goal["verdict"] != "unsure" and not goal["evidence_ids"]:
        raise ValueError("A definite verdict requires evidence.")
    if not isinstance(review.get("steps"), list):
        raise ValueError("Step sequence must be a list.")
    for step in review["steps"]:
        if step["name"] not in STEPS or not step.get("evidence_ids") or not set(step["evidence_ids"]) <= ids:
            raise ValueError("Step requires an allowed label and valid evidence.")
    return review


def validate_contract_review(review: dict, frames: dict[int, float], contract: dict, duration: float) -> dict:
    validate_review(review, frames)
    if type(review.get("completion_required")) is not bool:
        raise ValueError("Completion policy must be explicit.")
    if not review["completion_required"] and contract["goal"]["required"]:
        if not contract.get("operator_request") or not review.get("completion_override_reason"):
            raise ValueError("Completion override requires an operator request and explanation.")
    ids = {item["id"] for item in review["evidence"]}
    def assessment(item):
        if (not isinstance(item, dict) or item.get("verdict") not in ("yes", "no", "unsure")
                or not isinstance(item.get("reason"), str) or not isinstance(item.get("evidence_ids"), list)
                or not set(item["evidence_ids"]) <= ids
                or item["verdict"] != "unsure" and not item["evidence_ids"]):
            raise ValueError("Invalid grounded rule assessment.")
    assessment(review.get("task_action_shown"))
    rules = {rule["id"]: rule for rule in contract["process"]}
    received = review.get("rules")
    if not isinstance(received, list) or len({r["rule_id"] for r in received}) != len(received):
        raise ValueError("Rule assessments must be unique.")
    # Some responses put known format/privacy requirements into this array.
    # Keep them separate: local requirement checks remain authoritative.
    requirement_ids = set(contract.get("requirements", {})) - set(rules)
    misplaced = [item for item in received if item["rule_id"] in requirement_ids]
    if misplaced:
        review["requirement_observations"] = misplaced
        received = [item for item in received if item["rule_id"] not in requirement_ids]
        review["rules"] = received
    for item in received:
        if item["rule_id"] not in rules:
            raise ValueError("Unknown contract rule.")
        if not rules[item["rule_id"]].get("required") and item.get("evidence_ids") == []:
            item["verdict"] = "unsure"
            item["reason"] = str(item.get("reason", "")) + "; optional assessment has no cited evidence"
        assessment(item)
    if not {key for key, rule in rules.items() if rule.get("required") and rule.get("enabled", True)} <= {r["rule_id"] for r in received}:
        raise ValueError("Required process rule assessment missing.")
    for interval in review.get("intervals", []):
        start, end = interval["start_s"], interval["end_s"]
        if (not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end))
                or not 0 <= start < end <= duration or interval.get("rule_id") not in rules
                or type(interval.get("confirmed")) is not bool
                or not interval.get("evidence_ids") or not set(interval["evidence_ids"]) <= ids):
            raise ValueError("Invalid exclusion interval evidence.")
    requests = review.get("follow_up_requests", [])
    if not isinstance(requests, list) or len(requests) > MAX_FOLLOWUPS:
        raise ValueError("Too many follow-up requests.")
    for item in requests:
        start, end = item["start_s"], item["end_s"]
        if (not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end))
                or not 0 <= start < end <= duration or end - start > MAX_WINDOW_S):
            raise ValueError("Follow-up interval outside budget or clip.")
    return review


def followup_frames(rec, table: pd.DataFrame, requests: list[dict], destination: str) -> dict[int, float]:
    times = table.t.to_numpy()
    indices = set()
    for request in requests[:MAX_FOLLOWUPS]:
        targets = np.arange(request["start_s"], request["end_s"], 1 / FOLLOWUP_HZ)
        indices.update(int(index) for index in np.searchsorted(times, targets) if index < len(table))
    indices = sorted(indices)
    if len(indices) > MAX_FOLLOWUP_FRAMES:
        indices = [indices[i] for i in np.linspace(0, len(indices) - 1, MAX_FOLLOWUP_FRAMES).round().astype(int)]
    frames = {}
    root = Path(destination) / "followup"
    root.mkdir(parents=True, exist_ok=True)
    for row, frame in iter_sampled_frames(rec, table.iloc[indices]):
        path = root / f"frame_{row:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise ValueError("Could not save follow-up evidence.")
        table.at[row, "thumbnail_path"] = str(path.resolve())
        frames[int(row)] = float(table.at[row, "t"])
    return frames


def _request(client, prompt, contract, table, frames, model, stage, version):
    from truetake.measured import context
    measured_context = context(table, contract) if version == 2 else {}
    duration = float(table.sample_end.iloc[-1])
    plan = table.iloc[0].get("evidence_plan")
    focused = isinstance(plan, dict) and plan.get("mode") == "focused"
    values = {"{contract_yaml}": yaml.safe_dump(contract), "{duration_s}": str(duration),
              "{frames}": json.dumps([{"frame_id": row, "t": time} for row, time in frames.items()]),
              "{detector_evidence}": json.dumps(plan) if focused else "Detectors are fallible; brief hand exits are not failures.",
              "{overview_or_targeted_follow_up}": stage}
    for name, value in values.items():
        prompt = prompt.replace(name, value)
    prompt += "\nMeasured context (formula judges measured rules; you judge the goal): " + json.dumps(measured_context)
    content = [{"type": "text", "text": prompt if focused else prompt + "\nContract: " + json.dumps(contract)}]
    for row, time in frames.items():
        data = base64.b64encode(Path(table.at[row, "thumbnail_path"]).read_bytes()).decode()
        content.extend([{"type": "text", "text": f"Frame {row}, time {time:.6f} seconds"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}])
    response = client.messages.create(model=model, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": content}])
    table.attrs.setdefault("claude_call_usage", []).append({"input_tokens": response.usage.input_tokens,
                                                          "output_tokens": response.usage.output_tokens,
                                                          "images": len(frames)})
    if response.stop_reason == "max_tokens":
        raise ValueError("Review truncated; no verdict applied.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if table.attrs.get("review_debug_dir"):
        # Only model output is saved here; credentials are never part of the request content.
        Path(table.attrs["review_debug_dir"], f"claude_response_{stage}.txt").write_text(text, encoding="utf-8")
    review = bound_followups(parse_review(text))
    if version == 2:
        validate_contract_review(review, frames, contract, duration)
    else:
        validate_review(review, frames)
    return review, response.usage.input_tokens, response.usage.output_tokens


def detect_claude(table: pd.DataFrame, cache_dir: str, prompt_path: str, model: str = MODEL,
                  rec=None, evidence_dir: str | None = None) -> pd.DataFrame:
    table = table.copy()
    table.attrs["claude_call_usage"] = []
    table["claude_usage"] = pd.Series([None] * len(table), index=table.index, dtype=object)
    if evidence_dir is not None:
        table.attrs["review_debug_dir"] = evidence_dir
    table["claude_status"] = pd.Series([None] * len(table), index=table.index, dtype=object)
    table["claude_review"] = pd.Series([None] * len(table), index=table.index, dtype=object)
    first = table.index[0]
    plan = table.iloc[0].get("evidence_plan")
    focused = isinstance(plan, dict) and plan.get("mode") == "focused"
    frames = {int(row): float(sample.t) for row, sample in table.iterrows()
              if isinstance(sample.get("thumbnail_path"), str) and (not focused or bool(sample.get("claude_selected", False)))}
    def finish():
        calls = table.attrs.get("claude_call_usage", [])
        table.at[first, "claude_usage"] = {"mode": "focused" if focused else "baseline",
            "initial_images": len(initial_ids), "actual_api_calls": len(calls),
            "actual_images_sent": sum(call["images"] for call in calls),
            "actual_input_tokens": sum(call["input_tokens"] for call in calls),
            "actual_output_tokens": sum(call["output_tokens"] for call in calls),
            "usage_complete": not str(table.at[first, "claude_status"]).startswith("error:")}
        return table
    initial_ids = list(frames)
    if not frames:
        table.at[first, "claude_status"] = "no_evidence_frames"
        return finish()
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    contract = table.iloc[0].contract or {}
    version = contract.get("schema_version", 1)
    payload = prompt + json.dumps(contract, sort_keys=True) + model
    if focused:
        payload += json.dumps(plan, sort_keys=True)
    if rec is not None:
        with Path(rec.video_path).open("rb") as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        payload += digest.hexdigest()
    for row, time in frames.items():
        image = Path(table.at[row, "thumbnail_path"]).read_bytes()
        payload += f"{row}:{time}:" + hashlib.sha256(image).hexdigest()
    path = Path(cache_dir) / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"
    try:
        if path.exists():
            review = json.loads(path.read_text())
            cached_frames = {int(row): float(table.at[int(row), "t"]) for row in review.get("provenance", {}).get("frame_ids", frames)}
            if version == 2:
                validate_contract_review(review, cached_frames, contract, float(table.sample_end.iloc[-1]))
            else:
                validate_review(review, cached_frames)
            provenance = review.get("provenance", {})
            if evidence_dir is not None:
                missing = False
                for row in provenance.get("followup_frame_ids", []):
                    image_path = Path(evidence_dir) / "followup" / f"frame_{int(row):06d}.jpg"
                    if image_path.is_file():
                        table.at[int(row), "thumbnail_path"] = str(image_path.resolve())
                    else:
                        missing = True
                if missing and rec is not None:
                    followup_frames(rec, table, provenance.get("followup_requests", []), evidence_dir)
            table.at[first, "claude_status"] = "cached"
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            table.at[first, "claude_status"] = "missing_api_key"
            return finish()
        else:
            with anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=1) as client:
                review, input_tokens, output_tokens = _request(client, prompt, contract, table, frames, model, "overview", version)
                if focused:
                    review = route_followups(review, contract)
                overview = review
                requests = review.get("follow_up_requests", []) if version == 2 else []
                followup_ids = []
                if requests and rec is not None and evidence_dir is not None:
                    additional = followup_frames(rec, table, requests, evidence_dir)
                    followup_ids = list(additional)
                    frames.update(additional)
                    followup_prompt = prompt + "\nPrevious overview: " + json.dumps(overview)
                    review, used_input, used_output = _request(client, followup_prompt, contract, table, frames, model, "targeted_follow_up", version)
                    input_tokens += used_input
                    output_tokens += used_output
                review["provenance"] = {"model": model, "prompt_version": Path(prompt_path).stem,
                                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                                        "frame_ids": list(frames), "followup_frame_ids": followup_ids,
                                        "followup_requests": requests, "api_calls": 2 if followup_ids else 1,
                                        "initial_frame_ids": initial_ids,
                                        "images_sent": len(initial_ids) + (len(frames) if followup_ids else 0),
                                        "review_mode": "focused" if focused else "baseline"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(review, indent=2), encoding="utf-8")
            table.at[first, "claude_status"] = "ok"
        table.at[first, "claude_review"] = review
    except (anthropic.APIError, ValueError, KeyError, TypeError) as error:
        table.at[first, "claude_status"] = f"error:{type(error).__name__}"
    return finish()
