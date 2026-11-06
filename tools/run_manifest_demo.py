"""Execute manifest entries with task contracts; keep work products out of demo/."""

import argparse
import csv
import fcntl
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from truetake.batch import run_batch
from truetake.report import write_json
from truetake.value import analyze_batch
from truetake.viz_rerun import save_recording
from truetake.ingest import load_recording


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/demo_work"
RESULTS = ROOT / "outputs/manifest_run"
PROFILE = ROOT / "outputs/profiles/gopro_epic_calibrated.json"
FOLD = ROOT / "contracts/mercor_folding.yaml"
WASH = ROOT / "contracts/wash_dishes.yaml"


def read_entries() -> list[dict]:
    rows = []
    for path in [ROOT / "data/manifest.csv", ROOT / "data/clips/variants/manifest.csv"]:
        with path.open(newline="") as handle:
            rows += list(csv.DictReader(handle))
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest clip IDs must be unique")
    return rows


def stage_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError("Staging symlink points at a different source")
        return
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def seed_caches(output: Path, entries: list[dict]) -> None:
    old = ROOT / "outputs/demo"
    cache = old / "claude_cache"
    if cache.is_dir():
        shutil.copytree(cache, output / "claude_cache", dirs_exist_ok=True)
    for row in entries:
        source, target = old / row["clip_id"], output / row["clip_id"]
        if not source.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in ("detectors.onepass.json", "detectors.onepass.sha256"):
            if (source / name).is_file() and not (target / name).exists():
                shutil.copy2(source / name, target / name)
        if (source / "followup").is_dir():
            shutil.copytree(source / "followup", target / "followup", dirs_exist_ok=True)


def save_result(row: dict, card: dict | None, errors: list) -> dict:
    """Atomically publish the completion marker only after per-clip artifacts finish."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if Path(row["clip_id"]).name != row["clip_id"] or row["clip_id"] in ("", ".", "..", "batch"):
        raise ValueError("Clip ID must be a single filename and cannot be 'batch'")
    payload = {"schema_version": 1, "clip_id": row["clip_id"], "manifest_entry": row,
               "status": "complete" if card is not None else "failed", "card": card, "errors": errors}
    path = RESULTS / (row["clip_id"] + ".json")
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)
    return payload


def process_one(row: dict) -> dict:
    if row["attack"] == "controlled_reupload":
        original = row["clip_id"].removesuffix("_controlled_reupload")
        previous = RESULTS / (original + ".json")
        if not previous.is_file():
            raise ValueError("Controlled duplicate's original must finish first")
        payload = json.loads(previous.read_text())
        if not payload.get("card"):
            raise ValueError("Controlled duplicate's original failed")
        folder = WORK / "aliases" / row["clip_id"]
        folder.mkdir(parents=True, exist_ok=True)
        card = {**payload["card"], "clip_id": row["clip_id"],
                "fixture_kind": "CONTROLLED DEMO FIXTURE: identical source; not independent training footage",
                "card_path": str(folder / "card.json"), "table_path": str(folder / "frames.json"),
                "rerun_path": str(folder / "timeline.rrd"),
                "export": {"clip_id": row["clip_id"], "export_status": "not_exported", "eligible_for_training": False,
                           "reason": "Controlled duplicate; retain original"}}
        if not Path(card["table_path"]).is_file():
            shutil.copy2(payload["card"]["table_path"], card["table_path"])
        card["claude_usage"] = {**(card.get("claude_usage") or {}), "actual_api_calls": 0,
                                "actual_images_sent": 0, "actual_input_tokens": 0, "actual_output_tokens": 0}
        if not Path(card["rerun_path"]).is_file():
            rec = replace(load_recording(card["video_path"]), clip_id=row["clip_id"])
            table = pd.DataFrame(json.loads(Path(card["table_path"]).read_text()))
            save_recording(rec, table, card, card["rerun_path"])
    else:
        folding = row["task_expected"] == "folding" or row["source"] == "hub"
        group = ("hub_" if row["source"] == "hub" else "") + ("folding" if folding else "washing")
        stage, output = WORK / "staged" / group, WORK / "runs" / group
        output.mkdir(parents=True, exist_ok=True)
        video = stage / (row["clip_id"] + ".mp4")
        stage_link(ROOT / row["video_path"], video)
        if row["sensors_path"] and (ROOT / row["sensors_path"]).is_file():
            stage_link(ROOT / row["sensors_path"], video.with_name(video.stem + "_sensors.csv"))
        seed_caches(output, [row])
        result = run_batch(str(stage), str(FOLD if folding else WASH), str(output),
                           detector_mode="onepass", export=True, gyro_profile=str(PROFILE),
                           clip_ids=[row["clip_id"]], render=False)
        card = result["cards"].get(row["clip_id"])
        if card is None:
            return save_result(row, None, result["analysis"]["errors"])
        card["task_contract"] = str(FOLD if folding else WASH)
        errors = result["analysis"]["errors"]
    card.update(source=row["source"], split=row["split"], manifest_entry=row)
    write_json(card["card_path"], card)
    return save_result(row, card, errors if row["attack"] != "controlled_reupload" else [])


def run() -> dict:
    """No report aggregation here. Completed result files skip all processing."""
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    processed, skipped, missing, failed = [], [], [], []
    for row in read_entries():
        path = RESULTS / (row["clip_id"] + ".json")
        if path.is_file():
            # Do not overwrite a corrupt completion marker or silently rerun paid work.
            previous = json.loads(path.read_text())
            if previous.get("clip_id") != row["clip_id"]:
                raise ValueError(f"Result identity mismatch: {path}")
            print("SKIP RESULT", row["clip_id"], flush=True)
            skipped.append(row["clip_id"])
            continue
        if not (ROOT / row["video_path"]).is_file():
            missing.append({"clip_id": row["clip_id"], "source": row["source"], "reason": "video file missing"})
            print("SKIP MISSING", row["clip_id"], flush=True)
            continue
        print("START", row["clip_id"], flush=True)
        try:
            result = process_one(row)
        except (OSError, ValueError, RuntimeError) as error:
            result = save_result(row, None, [{"clip_id": row["clip_id"], "error": str(error)}])
        processed.append(row["clip_id"])
        if result["status"] == "failed" or result.get("errors"):
            failed.append(row["clip_id"])
        print("SAVED RESULT", row["clip_id"], result["status"], flush=True)
    summary = {"processed": processed, "skipped_results": skipped, "missing": missing, "failed": failed}
    write_json(WORK / "run_status.json", summary)
    print("Per-clip run finished. Rebuild reports separately with tools/build_demo_report.py", flush=True)
    return summary


def assemble_results() -> dict:
    """Read only per-clip results and saved tables; no inference or video decoding."""
    cards, tables, errors, missing = {}, {}, [], []
    entries = read_entries()
    for row in entries:
        path = RESULTS / (row["clip_id"] + ".json")
        if not path.is_file():
            missing.append({"clip_id": row["clip_id"], "source": row["source"],
                            "reason": "video file missing" if not (ROOT / row["video_path"]).is_file() else "not processed yet"})
            continue
        result = json.loads(path.read_text())
        errors.extend(result.get("errors", []))
        if result.get("card") is not None:
            cards[row["clip_id"]] = result["card"]
            tables[row["clip_id"]] = pd.DataFrame(json.loads(Path(result["card"]["table_path"]).read_text()))
    analysis = analyze_batch(tables, cards)
    analysis.update(errors=errors, skipped=missing, manifest_entries=len(entries), gyro_profile=str(PROFILE),
                    note="Saved per-clip results; missing or pending entries are not fabricated verdicts.")
    result = {"cards": cards, "analysis": analysis}
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "batch.json", result)
    write_json(WORK / "all_results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true", help="Build batch.json and reports from saved results only")
    args = parser.parse_args()
    if args.finalize:
        from build_demo_report import main
        main()
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(2, "Another manifest run is already active.\n")
            status = run()
            if status["failed"]:
                raise SystemExit(2)
