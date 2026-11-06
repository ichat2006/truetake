"""Publish saved results as a portable public demo and a separate internal view."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image, ImageFilter

from truetake.contracts import load_contract
from truetake.export import write_collection_manifest
from truetake.preview import browser_preview
from truetake.presentation import badge, plain_reason
from truetake.report import write_batch_report, write_json
from truetake.value import analyze_batch, suggest_next
from truetake.viz_rerun import save_batch_map


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/demo_work/all_results.json"
THUMBNAIL_BLUR_RADIUS = 24


def copy_asset(source: str, target: Path) -> str:
    source = Path(source).resolve()
    if source == target.resolve():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (not target.is_file() or target.stat().st_size != source.stat().st_size
            or target.stat().st_mtime_ns != source.stat().st_mtime_ns):
        shutil.copy2(source, target)
    return str(target)


def publish_subset(saved: dict, target: Path, public: bool) -> str:
    target.mkdir(parents=True, exist_ok=True)
    cards = deepcopy({name: card for name, card in saved["cards"].items()
                      if not public or card.get("source") != "hub"})
    tables = {}
    for name, card in cards.items():
        directory = target / name
        directory.mkdir(exist_ok=True)
        tables[name] = pd.DataFrame(json.loads(Path(card["table_path"]).read_text()))
        card["table_path"] = copy_asset(card["table_path"], directory / "frames.json")
        if Path(card["rerun_path"]).is_file():
            card["rerun_path"] = copy_asset(card["rerun_path"], directory / "timeline.rrd")
        card["card_path"] = str(directory / "card.json")
        preview = card.get("preview_path")
        if not preview or not Path(preview).is_file():
            preview = browser_preview(card["video_path"], str(target / "previews"), card.get("privacy_blur_intervals", []))
        card["preview_path"] = copy_asset(preview, target / "media" / (name + "_original.mp4"))
        card["presentation_preview_ready"] = True
        for index, image in enumerate(card.get("thumbnails", [])):
            destination = directory / "evidence" / f"{index:04d}.jpg"
            needs_blur = any(a <= image["t"] <= b for a, b in card.get("privacy_blur_intervals", []))
            if needs_blur and not image.get("unredacted_path"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(image["path"]) as source_image:
                    source_image.filter(ImageFilter.GaussianBlur(THUMBNAIL_BLUR_RADIUS)).save(destination)
                image["path"] = str(destination)
            else:
                image["path"] = copy_asset(image["path"], destination)
        exported = card.get("export") or {}
        for key in ("video_path", "frame_map_path", "sensors_path", "manifest_path"):
            if exported.get(key):
                exported[key] = copy_asset(exported[key], target / "exports" / name / Path(exported[key]).name)
        if exported.get("manifest_path"):
            write_json(exported["manifest_path"], exported)
        write_json(card["card_path"], card)
    analysis = analyze_batch(tables, cards)
    included = set(cards)
    analysis["errors"] = [error for error in saved["analysis"].get("errors", []) if error["clip_id"] in included]
    analysis["skipped"] = [row for row in saved["analysis"].get("skipped", []) if not public or row.get("source") != "hub"]
    wanted = load_contract(str(ROOT / "contracts/mercor_folding.yaml"))["value"]["wanted_variations"]
    analysis["film_next"] = suggest_next(cards, wanted)
    analysis["scope"] = "Public-facing: Hub excluded" if public else "Internal evaluation; includes any available Hub clips"
    analysis["gyro_profile"] = str(ROOT / "outputs/profiles/gopro_epic_calibrated.json")
    collection = write_collection_manifest({name: card.get("export", {}) for name, card in cards.items()}, str(target / "exports"))
    analysis["exports"] = {key: value for key, value in collection.items() if key != "entries"}
    save_batch_map(analysis, str(target / "batch_map.rrd"))
    report = write_batch_report(cards, analysis, str(target))
    with (target / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "source", "verdict", "reason", "usable_seconds", "gyro_result"])
        writer.writeheader()
        for name, card in cards.items():
            gyro = (card.get("verification") or {}).get("passed")
            writer.writerow({"clip_id": name, "source": card.get("source", "unknown"), "verdict": badge(card),
                             "reason": plain_reason(card), "usable_seconds": round(card.get("usable_seconds", 0.), 2),
                             "gyro_result": "PASS" if gyro is True else "FAIL" if gyro is False else "NOT VERIFIED"})
    with (target / "skipped.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "source", "reason"])
        writer.writeheader()
        writer.writerows(analysis["skipped"])
    return report


def main() -> None:
    from run_manifest_demo import assemble_results
    saved = assemble_results()
    print("Public:", publish_subset(saved, ROOT / "outputs/demo", True), flush=True)
    print("Internal:", publish_subset(saved, ROOT / "outputs/demo_internal", False), flush=True)


if __name__ == "__main__":
    main()
