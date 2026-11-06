"""Validate operator-supplied lab contracts before any detector or API work."""

import math
from pathlib import Path

import yaml


SUPPORTED_VERSIONS = {1, 2}


def load_contract(path: str, instructions: str = "") -> dict:
    source = Path(path)
    if source.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Task contract must be a YAML file.")
    contract = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(contract.get("task"), str) or not contract["task"].strip():
        raise ValueError("Contract must contain a nonempty task string.")
    if contract.get("draft"):
        raise ValueError("Draft contract is for review, not execution.")
    version = contract.get("schema_version", 1)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("Unsupported contract schema version.")
    if version == 2:
        if not isinstance(contract.get("goal"), dict) or not isinstance(contract["goal"].get("required"), bool):
            raise ValueError("Version-2 contract requires a goal with a boolean required field.")
        rules = contract.get("process")
        if not isinstance(rules, list) or not rules or any(not isinstance(rule, dict) or not isinstance(rule.get("id"), str) for rule in rules):
            raise ValueError("Contract process must contain rules with string IDs.")
        ids = [rule["id"] for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Process rule IDs must be unique.")
        for rule in rules:
            for key in ("target", "review_band"):
                if key in rule and (type(rule[key]) not in (int, float) or not math.isfinite(rule[key]) or not 0 <= rule[key] <= 1):
                    raise ValueError(f"{key} must be a fraction in [0,1].")
            if not isinstance(rule.get("required", False), bool):
                raise ValueError("Rule required fields must be boolean.")
            for key in ("brief_gap_tolerance_s", "max_idle_s"):
                value = rule.get(key)
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError(f"{key} must be nonnegative finite seconds.")
        minimum = contract.get("duration", {}).get("min_usable_seconds")
        if minimum is not None and (not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or minimum < 0):
            raise ValueError("Minimum duration must be null or nonnegative finite seconds.")
        if instructions:
            contract["operator_request"] = instructions
    elif instructions:
        raise ValueError("Free-text collection instructions require a version-2 contract.")
    return contract
