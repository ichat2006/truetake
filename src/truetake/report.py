"""Write clip cards and an offline, clickable batch report."""

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter
from truetake.preview import browser_preview

from truetake.models import CheckResult


def build_card(clip_id: str, table: pd.DataFrame, checks: list[CheckResult], quality: dict) -> dict:
    evidence = []
    if "thumbnail_path" in table:
        evidence = [{"t": float(row.t), "path": row.thumbnail_path} for _, row in table.iterrows()
                    if isinstance(row.thumbnail_path, str)]
    verification = next((asdict(check) for check in checks if check.name == "verification.gyro"), None)
    return {"schema_version": 1, "clip_id": clip_id, **quality, "checks": [asdict(check) for check in checks],
            "duration_s": float(table.sample_end.iloc[-1]), "thumbnails": evidence,
            "claude_status": table.iloc[0].get("claude_status", "not_run"),
            "claude_review": table.iloc[0].get("claude_review"),
            "claude_usage": table.iloc[0].get("claude_usage"),
            "evidence_plan": table.iloc[0].get("evidence_plan"),
            "detector_stats": table.iloc[0].get("detector_stats"),
            "verification": verification, "sensor_present": bool(table.sensor_present.any())}


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_batch_report(cards: dict[str, dict], batch: dict, output_dir: str) -> str:
    root = Path(output_dir).resolve()
    protected = (Path(__file__).resolve().parents[2] / "data").resolve()
    if root == protected or protected in root.parents:
        raise ValueError("Reports must be written outside data/.")
    root.mkdir(parents=True, exist_ok=True)
    usage = [card.get("claude_usage") or {} for card in cards.values()]
    batch["claude_usage"] = {key: sum(item.get(key, 0) for item in usage) for key in
        ("actual_api_calls", "actual_images_sent", "actual_input_tokens", "actual_output_tokens")}
    for card in cards.values():
        from truetake.licenses import apply_license
        card.update(apply_license(card))
        if not card['license_permissions']['modify']:
            from truetake.export import omitted_entry
            card['export']=omitted_entry(card)
        spans = card.get("privacy_blur_intervals", [])
        if not card['media_display_allowed']:
            card.update(media_hidden=True,thumbnails=[],preview_path='')
            continue
        if not card['license_permissions']['modify'] and not card.get('presentation_preview_ready'):
            card['preview_path']=card['video_path']
            card['presentation_preview_ready']=True
        if not card.get("presentation_preview_ready") or not Path(card.get("preview_path", "")).is_file():
            card["preview_path"] = browser_preview(card["video_path"], str(root / "previews"), spans)
        card["privacy_preview_blurred"] = bool(spans)
        for item in card["thumbnails"]:
            if not card.get("presentation_preview_ready") and any(a <= item["t"] <= b for a, b in spans):
                destination = root / "redacted_thumbnails" / f"{card['clip_id']}_{item['t']:.6f}.jpg"
                destination.parent.mkdir(exist_ok=True)
                item.setdefault("unredacted_path", item["path"])
                with Image.open(item["unredacted_path"]) as image:
                    image.filter(ImageFilter.GaussianBlur(24)).save(destination)
                item["path"] = str(destination)
        if Path(card["card_path"]).is_file():
            write_json(card["card_path"], card)
    write_json(root / "batch.json", {"cards": cards, "analysis": batch})
    from truetake.presentation import render_report
    path = root / "index.html"
    path.write_text(render_report(cards, batch, root), encoding="utf-8")
    return str(path)
