"""Contract-driven measurement summaries and target bands; no task limits in code."""
from pathlib import Path
import math

import numpy as np
import pandas as pd
import yaml

DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "contracts/defaults.yaml"
HAND_HOLD_S = 0.21  # Engineering limit matching the 5 Hz detector schedule.
TIME_EPSILON = 1e-6

def defaults() -> dict:
    return yaml.safe_load(DEFAULTS_PATH.read_text())

def band_result(value: float | None, target: float, band: float, cap: float | None = None,
                enabled: bool = True) -> dict:
    lower = target - band if enabled else target
    upper = (target + band if cap is None else min(target + band, max(target, cap))) if enabled else target
    state = "review" if value is None else "fail" if value < lower - TIME_EPSILON else "review" if value < upper - TIME_EPSILON else "pass"
    return {"value": value, "target": target, "band": band, "lower": lower, "upper": upper, "state": state}

def ranges(table: pd.DataFrame, mask: np.ndarray) -> list[list[float]]:
    found = []
    for i in np.flatnonzero(mask):
        a, b = float(table.t.iloc[i]), float(table.sample_end.iloc[i])
        if found and a <= found[-1][1] + TIME_EPSILON:
            found[-1][1] = b
        else:
            found.append([a, b])
    return found

def context(table: pd.DataFrame, contract: dict | None = None, excluded: list | None = None) -> dict:
    contract = contract or table.iloc[0].get("contract") or {}
    settings = {**defaults(), **{k:v for k,v in contract.items() if k in defaults()}}
    engineering = {**defaults()["measurement"], **contract.get("measurement", {})}
    weights = table.sample_end.to_numpy() - table.t.to_numpy()
    active = np.ones(len(table), dtype=bool)
    for a, b in excluded or []:
        active &= ~((table.t >= a - TIME_EPSILON) & (table.sample_end <= b + TIME_EPSILON))
    denominator = float(weights[active].sum())
    rules = []
    for rule in contract.get("process", []):
        if "target" not in rule or not rule.get("enabled", True):
            continue
        signal = np.full(len(table), np.nan)
        kind = rule.get("measurement")
        column = "hand_count" if kind == "hand_present" else "hands_inside"
        if kind in ("hands_inside", "hand_present") and column in table and "people_status" in table:
            scheduled = table[table.people_status != "not_sampled"]
            if len(scheduled):
                indices = np.searchsorted(scheduled.t.to_numpy(), table.t.to_numpy(), side="right") - 1
                selected = np.maximum(indices, 0)
                good = (indices >= 0) & (table.t.to_numpy() - scheduled.t.to_numpy()[selected] <= HAND_HOLD_S + TIME_EPSILON)
                good &= scheduled.people_status.to_numpy()[selected] == "ok"
                values = scheduled[column].to_numpy(dtype=float)
                if kind == "hand_present":
                    values = np.where(np.isfinite(values), (values >= 1).astype(float), np.nan)
                signal[good] = values[selected[good]]
        elif kind == "active_fraction" and "frame_change" in table:
            signal = np.where(table.frame_change.notna(), 1., np.nan)
            idle = table.frame_change.to_numpy() <= engineering["idle_change_max"]
            for a, b in ranges(table, idle):
                if b-a > rule["max_idle_s"]:
                    signal[(table.t >= a) & (table.t < b)] = 0.
        known = active & np.isfinite(signal)
        support = float(weights[known].sum()) / denominator if denominator else 0.
        coverage = float(np.sum(weights[known]*signal[known])/weights[known].sum()) if support >= engineering["min_observed_fraction"] and weights[known].sum() else None
        assessment = band_result(coverage, rule["target"], rule.get("review_band", settings["review_band"]),
                                 settings["review_upper_cap"], settings["close_calls_enabled"])
        rules.append({"rule_id": rule["id"], "measurement": kind, "observed_fraction": support,
                      "flags": ranges(table, active & (signal < 1)), **assessment})
    minimum = contract.get("min_usable_seconds", contract.get("duration", {}).get("min_usable_seconds"))
    seconds = None if minimum is None else band_result(denominator, minimum,
        contract.get("review_band_seconds", settings["review_band_seconds"]), enabled=settings["close_calls_enabled"])
    return {"rules": rules, "usable_seconds": denominator, "usable_assessment": seconds,
            "close_calls_enabled": settings["close_calls_enabled"]}
