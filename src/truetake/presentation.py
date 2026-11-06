"""Plain-language presentation of saved verdicts; never changes pipeline decisions."""

from collections import Counter
import html
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from truetake.verify_gyro import prepare_signals, _interpolate


MAP_WIDTH = 1200
MAP_HEIGHT = 520
MAX_REASON_CHARACTERS = 220
MAX_TRACE_POINTS = 180
COLORS = {"APPROVED": "#46d6a0", "CUT": "#64c4ff", "REJECTED": "#ff8581",
          "DUPLICATE": "#b5a2ff", "NOT VERIFIED": "#f4c777"}


def badge(card: dict) -> str:
    if card.get("fixture_kind"):
        return "DUPLICATE"
    if card["status"] == "REJECT":
        return "REJECTED"
    if card["status"] != "ACCEPT":
        return "NOT VERIFIED"
    cuts = (card.get("export") or {}).get("plan", {}).get("applied_cuts", [])
    return "CUT" if cuts else "APPROVED"


def plain_reason(card: dict) -> str:
    verdict = badge(card)
    gyro = card.get("verification") or {}
    if verdict == "DUPLICATE":
        return "This is the same recording as its original; keep one copy."
    if gyro.get("passed") is False:
        return "Phone motion doesn't match the video."
    if verdict == "CUT":
        return "Meets the task checklist after unusable sections are removed."
    if verdict == "APPROVED":
        return "Meets the task checklist."
    reasons = card.get("decision_reasons", [])
    if not reasons:
        reasons = [check["reason"] for check in card.get("checks", []) if check.get("passed") is not True]
    text = reasons[0] if reasons else "There is not enough evidence to verify this clip."
    text = re.sub(r"^(?:goal|process|requirements)\.[\w_]+:\s*", "", text)
    text = re.sub(r"\s*\(frames?[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.split(r"(?<=[.!?])\s+", text)[0]
    if len(text) > MAX_REASON_CHARACTERS:
        text = text[:MAX_REASON_CHARACTERS].rsplit(" ", 1)[0] + "…"
    return text


def reason_time(card: dict) -> float:
    for check in card.get("checks", []):
        if card["status"] != "ACCEPT" and check.get("passed") is False and check.get("flags"):
            return float(check["flags"][0][0])
    review = card.get("claude_review") or {}
    ids = set((review.get("goal") or {}).get("evidence_ids", []))
    return next((float(item["t"]) for item in review.get("evidence", []) if item["id"] in ids), 0.0)


def timestamp(seconds: float) -> str:
    return f"{int(seconds)//60}:{int(seconds)%60:02d}"


def motion_plot(card: dict, root: Path) -> str | None:
    if card.get("clip_id") == "splice_P03_109_P04_104":
        evidence = Path(__file__).resolve().parents[2] / "outputs/splice_gyro/windows.csv"
        if evidence.is_file():
            windows = pd.read_csv(evidence)
            points = " ".join(f"{12+456*(row.start_s+2.5)/60:.2f},{95-80*row.score:.2f}" for row in windows.itertuples())
            folder = root / "motion"
            folder.mkdir(exist_ok=True)
            target = folder / (card["clip_id"] + ".svg")
            target.write_text("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 120'><rect width='480' height='120' fill='#111b28'/>"
                              + f"<polyline points='{points}' fill='none' stroke='#64c4ff' stroke-width='2'/>"
                              + "<path d='M240 8V100' stroke='#ff8581' stroke-dasharray='4 3'/><text x='245' y='18' fill='#ff8581' font-size='11'>cut 0:30</text><text x='12' y='115' fill='#b7c6d5' font-size='11'>5-second window peak scores · evidence only</text></svg>")
            return str(target)
    alignment = card.get("gyro_alignment") or {}
    if not alignment.get("axis_mapping") or alignment.get("offset_s") is None:
        return None
    table_path = Path(card.get("table_path", ""))
    if not table_path.is_file():
        return None
    table = pd.DataFrame(json.loads(table_path.read_text()))
    if not {"flow_pan", "gyro_gx", "flow_t", "flow_status"} <= set(table):
        return None
    grid, videos = prepare_signals(table)
    if not len(grid) or not alignment.get("axis_scores"):
        return None
    axis = max(alignment["axis_scores"], key=alignment["axis_scores"].get)
    pair = alignment["axis_mapping"][axis]
    video = videos[("flow_pan", "flow_tilt", "flow_roll").index(axis)]
    gyro = _interpolate(table.t.to_numpy(), table[pair["axis"]].to_numpy(), grid + alignment["offset_s"]) * pair["sign"]
    traces = []
    for values, color in [(video, "#64c4ff"), (gyro, "#f4c777")]:
        finite = np.isfinite(values)
        normalized = values.copy()
        if finite.sum() < 2:
            continue
        normalized[finite] = (values[finite] - values[finite].mean()) / max(float(values[finite].std()), 1e-8)
        indices = np.linspace(0, len(grid) - 1, min(MAX_TRACE_POINTS, len(grid))).round().astype(int)
        # Split at gaps so the plot never implies missing evidence is continuous.
        groups = np.split(indices, np.flatnonzero(~finite[indices]) + 1)
        for group in groups:
            group = group[finite[group]]
            if len(group) < 2:
                continue
            points = " ".join(f"{12+456*(grid[i]-grid[0])/max(float(np.ptp(grid)),.1):.2f},{58-15*np.clip(normalized[i],-3,3):.2f}" for i in group)
            traces.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='1.5'/>")
    folder = root / "motion"
    folder.mkdir(exist_ok=True)
    target = folder / (card["clip_id"] + ".svg")
    target.write_text("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 120' role='img' aria-label='Video motion and phone motion'><rect width='480' height='120' fill='#111b28'/><path d='M12 58H468' stroke='#334155'/>" + "".join(traces) + "<text x='12' y='112' fill='#64c4ff' font-size='11'>Video</text><text x='75' y='112' fill='#f4c777' font-size='11'>Phone motion</text></svg>")
    return str(target)


def render_report(cards: dict[str, dict], analysis: dict, root: Path) -> str:
    esc = html.escape

    def link(path):
        return esc(Path(os.path.relpath(path, root)).as_posix(), quote=True)

    counts = Counter(badge(card) for card in cards.values())
    approved = [card for card in cards.values() if badge(card) in ("APPROVED", "CUT")]
    minutes = sum(card.get("usable_seconds", 0.) for card in approved) / 60
    headline = (f"{len(cards)} clips in → {len(approved)} approved ({minutes:.1f} min usable) · "
                f"{counts['REJECTED']} rejected · {counts['DUPLICATE']} duplicates · {counts['NOT VERIFIED']} not verified")
    max_x = max([abs(p["x"]) for p in analysis.get("map", [])] + [.01])
    max_y = max([abs(p["y"]) for p in analysis.get("map", [])] + [.01])
    points, positions = [], []
    for point in analysis.get("map", []):
        clip_id = point["clip_id"]
        if clip_id not in cards:
            continue
        card = cards[clip_id]
        x, y = 560 + 470 * point["x"] / max_x, 225 - 180 * point["y"] / max_y
        # Offset overlapping thumbnails for inspection; retain approximate layout.
        attempts = 0
        while any(abs(x-a) < 78 and abs(y-b) < 55 for a, b in positions) and attempts < 40:
            attempts += 1
            x = min(1090, max(20, x + 80 * math.cos(attempts * 2.4)))
            y = min(440, max(15, y + 65 * math.sin(attempts * 2.4)))
        positions.append((x, y))
        thumbs = card.get("thumbnails", [])
        thumb = thumbs[len(thumbs)//2]["path"] if thumbs else None
        visual = f"<image href='{link(thumb)}' x='{x:.1f}' y='{y:.1f}' width='72' height='44' preserveAspectRatio='xMidYMid slice'/>" if thumb else ""
        color = COLORS[badge(card)]
        points.append(f"<a href='#{esc(clip_id)}'><title>{esc(clip_id)} · {badge(card)}</title><rect x='{x-2:.1f}' y='{y-2:.1f}' width='76' height='48' rx='5' fill='#1b293a' stroke='{color}' stroke-width='3'/>{visual}<text x='{x+36:.1f}' y='{y+58:.1f}' text-anchor='middle'>{esc(clip_id[:22])}</text></a>")
    placed = {point["clip_id"] for point in analysis.get("map", [])}
    unplaced = [name for name in cards if name not in placed]
    for index, clip_id in enumerate(unplaced):
        card = cards[clip_id]
        x, y = 30 + 100 * index, 455
        thumbs = card.get("thumbnails", [])
        image = f"<image href='{link(thumbs[0]['path'])}' x='{x}' y='{y}' width='72' height='44'/>" if thumbs else ""
        points.append(f"<a href='#{esc(clip_id)}'><title>{esc(clip_id)} · no comparison signal</title><rect x='{x-2}' y='{y-2}' width='76' height='48' fill='#152132' stroke='{COLORS[badge(card)]}' stroke-width='3'/>{image}<text x='{x}' y='{y-7}'>Unplaced</text></a>")
    panels = []
    for clip_id, card in cards.items():
        verdict, color = badge(card), COLORS[badge(card)]
        player, edited_player = "video_" + clip_id, "corrected_" + clip_id
        export = card.get("export") or {}
        gyro = card.get("verification") or {}
        moment = reason_time(card)
        cuts = export.get("plan", {}).get("applied_cuts", [])
        duration = max(float(card.get("duration_s", 0.)), .001)
        bars = "".join(f"<button class='cut' style='left:{100*cut['start_s']/duration:.3f}%;width:{100*(cut['end_s']-cut['start_s'])/duration:.3f}%' data-player='{esc(player)}' data-seek='{cut['start_s']:.6f}' title='Removed: {cut['start_s']:.1f}–{cut['end_s']:.1f} seconds'></button>" for cut in cuts)
        original = f"<video id='{esc(player)}' controls preload='metadata' src='{link(card['preview_path'])}'></video>"
        trimmed = f"<video id='{esc(edited_player)}' controls preload='metadata' src='{link(export['video_path'])}'></video>" if export.get("video_path") else "<div class='empty-player'>No approved output for this clip.</div>"
        edited_label = "Trimmed" if cuts else "Retained copy — no cuts" if export.get("video_path") else "Output"
        sensor_text = ("Phone motion matches the video." if gyro.get("passed") is True else
                       "Phone motion doesn't match the video." if gyro.get("passed") is False else
                       "Not verified: no motion sensor data" if not card.get("sensor_present") else
                       "Not verified: motion evidence is inconclusive")
        plot = motion_plot(card, root)
        sensor_visual = f"<img class='motion-plot' src='{link(plot)}' alt='Video motion compared with phone motion'>" if plot else ""
        if clip_id == "splice_P03_109_P04_104" and plot:
            sensor_text += " Match drops right after the cut at 0:30."
        checks = "".join(f"<li><b>{esc(check['name'])}</b>: {esc(str(check['passed']))} — {esc(check['reason'])}</li>" for check in card.get("checks", []))
        evidence = "".join(f"<figure><img loading='lazy' data-player='{esc(player)}' data-seek='{item['t']:.6f}' src='{link(item['path'])}'><figcaption>{timestamp(item['t'])}</figcaption></figure>" for item in card.get("thumbnails", []))
        downloads = " · ".join(f"<a href='{link(export[key])}'>{label}</a>" for key, label in
                               (("video_path", "Download video"), ("sensors_path", "Retimed sensors"), ("frame_map_path", "Frame timestamp map"), ("manifest_path", "Edit manifest")) if export.get(key))
        map_rows = "".join(f"<tr><td>{segment['source_start_s']:.3f}–{segment['source_end_s']:.3f}s</td><td>{segment['output_start_s']:.3f}–{segment['output_end_s']:.3f}s</td></tr>" for segment in export.get("plan", {}).get("kept_segments", []))
        technical = esc(json.dumps({key: card.get(key) for key in ("quality_score", "claude_status", "claude_usage", "claude_review", "manifest_entry", "gyro_alignment", "fixture_kind")}, indent=2))
        panels.append(f"""<article id='{esc(clip_id)}'><div class='card-heading'><h2>{esc(clip_id)}</h2><span class='badge' style='--badge:{color}'>{verdict}</span></div>
<p class='reason'>{esc(plain_reason(card))} <button class='time-link' data-player='{esc(player)}' data-seek='{moment:.6f}'>↗ {timestamp(moment)}</button></p>
<p class='usable'>{card.get('usable_seconds',0.):.1f} usable seconds <small>· estimated from sampled evidence</small></p>
<div class='comparison'><section><h3>Original</h3>{original}<div class='cut-bar' aria-label='Cut segments shaded'>{bars}</div><small>{'Shaded sections are removed from the output.' if cuts else 'No cuts applied.'}</small></section><section><h3>{edited_label}</h3>{trimmed}</section></div>
<div class='motion'>{sensor_visual}<p>{sensor_text}</p></div>
<details><summary>Technical details, scores and evidence</summary><p>Meets the task checklist: {card.get('quality_score',0.):.1f}/100 decided-rule agreement; not an acceptance probability.</p><p>How different from other clips: {analysis.get('scene_distinctness',{}).get(clip_id,0.):.1f}/100 relative appearance difference.</p><ul>{checks}</ul><div class='frames'>{evidence}</div><p>{downloads}</p>
<details><summary>Original → edited timestamp map</summary><table><tr><th>Original</th><th>Output</th></tr>{map_rows}</table></details>
<p><a href='{link(card['rerun_path'])}'>Rerun recording</a> · <a href='{link(card['card_path'])}'>Card JSON</a> · <a href='{link(card['table_path'])}'>Per-frame table</a></p>
<p>{esc(export.get('audio_policy',''))} {esc(export.get('sensor_policy',''))} {esc(export.get('post_edit_task_assessment',''))}</p><pre>{technical}</pre></details></article>""")
    legend = "".join(f"<span><i style='background:{color}'></i>{name.title()}</span>" for name, color in COLORS.items())
    ranking = "".join(f"<li>{esc(item['clip_id'])}: {esc(item['reason'])}</li>" for item in analysis.get("value_ranking", [])) or "<li>No approved clips available yet.</li>"
    diagnostics = esc(json.dumps({key: value for key, value in analysis.items() if key != "distance_matrix"}, indent=2))
    return document(headline, str(analysis.get("film_next", "Record a new task setup or technique to expand the collection.")),
                    legend, "".join(points), "".join(panels), ranking, diagnostics)


def document(headline: str, next_clip: str, legend: str, points: str, panels: str, ranking: str, diagnostics: str) -> str:
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>TrueTake — clip review</title><style>
:root{{color-scheme:dark;--ink:#edf3fa;--muted:#a9b8ca;--line:#2b394b}}*{{box-sizing:border-box}}body{{margin:0;background:#0b111b;color:var(--ink);font:16px system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:38px 28px 70px}}header{{padding:24px 0 36px}}.brand{{font-weight:800;letter-spacing:-.04em;font-size:26px}}.brand b{{color:#46d6a0}}h1{{font-size:clamp(30px,4.3vw,58px);line-height:1.14;max-width:1180px;font-weight:650;letter-spacing:-.04em;margin:25px 0}}.next{{font-size:19px;color:#c8e7d9}}h2{{font-size:21px;overflow-wrap:anywhere}}h3{{font-size:14px;color:var(--muted);font-weight:500}}p{{line-height:1.55}}a,.time-link{{color:#8dd4ff}}.box,article{{border:1px solid var(--line);background:#111c2a;border-radius:18px;padding:25px;margin-bottom:24px}}.map{{width:100%;display:block;background:radial-gradient(ellipse at center,#18283c,#0f1825);border-radius:12px;min-height:300px}}.map text{{font-size:10px;fill:#c7d4e5}}.legend{{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:var(--muted)}}.legend i{{width:9px;height:9px;display:inline-block;border-radius:3px;margin-right:7px}}.card-heading{{display:flex;align-items:center;justify-content:space-between;gap:20px}}.badge{{color:var(--badge);border:1px solid var(--badge);background:#0c1520;padding:7px 12px;border-radius:20px;font-size:12px;font-weight:750;white-space:nowrap}}.reason{{font-size:18px;max-width:1000px}}button{{font:inherit;cursor:pointer}}.time-link{{border:0;background:transparent;padding:0 6px}}.usable{{font-size:23px;font-weight:650;margin:14px 0 24px}}small{{font-size:12px;color:var(--muted);font-weight:400}}.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}video{{width:100%;display:block;aspect-ratio:16/9;background:#080c13;border-radius:9px}}.empty-player{{display:grid;place-content:center;aspect-ratio:16/9;border:1px dashed #36475c;border-radius:9px;color:var(--muted);padding:20px;text-align:center}}.cut-bar{{height:10px;position:relative;background:#267458;border-radius:6px;margin:12px 0 6px;overflow:hidden}}.cut{{position:absolute;height:100%;top:0;border:0;background:repeating-linear-gradient(135deg,#b14747,#b14747 4px,#6d3037 4px,#6d3037 8px)}}.motion{{display:flex;align-items:center;gap:22px;margin:24px 0 8px}}.motion p{{font-size:14px;color:var(--muted)}}.motion-plot{{width:280px;max-width:45%;border-radius:8px}}details{{margin:22px 0 4px;color:var(--muted)}}summary{{cursor:pointer;font-size:13px}}details p,li{{font-size:13px;line-height:1.6}}pre{{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:380px;overflow:auto;background:#0a121e;padding:18px;border-radius:8px}}.frames{{display:flex;gap:9px;overflow:auto;margin-top:18px}}figure{{margin:0;flex:none}}figure img{{width:140px;border-radius:6px;cursor:pointer}}figcaption{{font-size:11px}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:8px 18px;border-bottom:1px solid var(--line)}}footer{{font-size:12px;color:var(--muted)}}@media(max-width:760px){{main{{padding:20px 14px}}.comparison{{grid-template-columns:1fr}}.card-heading{{align-items:flex-start}}.box,article{{padding:17px}}.map{{min-height:220px}}.motion{{flex-direction:column;align-items:flex-start}}.motion-plot{{max-width:100%}}}}
</style></head><body><main><header><div class='brand'>True<b>Take</b></div><h1>{html.escape(headline)}</h1><p class='next'>Film next: {html.escape(next_clip)}</p></header>
<section class='box'><h2>How different from other clips</h2><p>Each point is one clip. Nearby thumbnails show similar scenes and task actions; the axes are not quality scores. Click a thumbnail to open its result.</p><svg class='map' viewBox='0 0 {MAP_WIDTH} {MAP_HEIGHT}' aria-label='Clip similarity map'>{points}</svg><div class='legend'>{legend}</div></section>
{panels}<details><summary>Collection details and provenance</summary><ul>{ranking}</ul><pre>{diagnostics}</pre></details><footer>TrueTake · task checklist, motion verification and collection value</footer></main>
<script>document.addEventListener('click',function(e){{const item=e.target.closest('[data-seek]');if(!item)return;const v=document.getElementById(item.dataset.player);if(!v)return;const seek=()=>{{v.currentTime=Number(item.dataset.seek)}};if(v.readyState>=1)seek();else v.addEventListener('loadedmetadata',seek,{{once:true}});v.scrollIntoView({{behavior:'smooth',block:'center'}});}});</script></body></html>"""
