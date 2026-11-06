"""Re-evaluate saved per-frame evidence only; never calls detectors or Claude."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from truetake.calibration import load_profile
from truetake.verify_gyro import measure_alignment, check_verification
from truetake.export import omitted_entry

ROOT = Path(__file__).resolve().parents[1]

def rescore_gyro():
    profile = load_profile(str(ROOT / "outputs/profiles/gopro_epic_calibrated.json"))
    rows, updates = [], []
    for path in (ROOT / "outputs/manifest_run").glob("*.json"):
        if path.name == "batch.json":
            continue
        payload = json.loads(path.read_text())
        card = payload.get("card") or {}
        if not card.get("sensor_present"):
            continue
        table = pd.DataFrame(json.loads(Path(card["table_path"]).read_text()))
        meta = dict(table.iloc[0]["clip_meta"] or {})
        meta["gyro_calibration"] = profile
        table.at[table.index[0], "clip_meta"] = meta
        table = measure_alignment(table)
        check = asdict(check_verification(table)[0])
        old = (card.get("verification") or {}).get("passed")
        rows.append({"clip_id": card["clip_id"], "source": card["source"], "split": card["split"],
                     "score": check["score"], "old_verdict": old, "new_verdict": check["passed"],
                     "close_call": table.iloc[0].gyro_alignment.get("close_call", False)})
        updates.append((path, payload, card, table, check))
    out = ROOT / "outputs/overnight/step1_gyro.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    failures = [row for row in rows if row["source"] != "variant" and row["split"] == "train" and row["new_verdict"] is False]
    if failures:
        print("STOP: real train clips failed", failures)
        raise SystemExit(2)
    for path, payload, card, table, check in updates:
        card["verification"] = check
        card["gyro_alignment"] = table.iloc[0].gyro_alignment
        card["checks"] = [item for item in card["checks"] if item["name"] != "verification.gyro"] + [check]
        if check["passed"] is False:
            card.update(status="REJECT", usable_seconds=0., approved_usable_seconds=0., usable_intervals=[])
            card["decision_reasons"] = [check["reason"]]
            card["export"] = omitted_entry(card)
        table.to_json(card["table_path"], orient="records", indent=2, double_precision=15)
        Path(card["card_path"]).write_text(json.dumps(card, indent=2))
        path.write_text(json.dumps(payload, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    rescore_gyro()
