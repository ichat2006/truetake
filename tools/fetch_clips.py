"""Plan and fetch a small, pinned corpus. Existing data files are immutable.

Prepare metadata: uv run python tools/fetch_clips.py prepare
One EPIC test:    uv run python tools/fetch_clips.py fetch --source epic --limit 1
Inventory:       uv run python tools/fetch_clips.py status
See docs/CLIP_SET.md for access gates and provenance limitations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import urlopen
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPIC_REPO = "awsaf49/epic_kitchens_100"
EPIC_REV = "c5e573b9295c06bd6d14bd5a9b9c7b178f00d7b1"
HUB_REPO = "Hubdata/egocentric-iphone-rgb"
HUB_REV = "a90d2d89eb1cd4ca824d23d4dfdddc4edf463e08"
HA_REPO = "humanarchive/HA-Multi-Samples"
ANNOTATION_REV = "ea8b40457a400c3fffa1c7f406ef3dc169cc2522"
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/"
    f"{ANNOTATION_REV}/EPIC_100_train.csv"
)
HF_COMMAND = ["uvx", "--from", "huggingface_hub==2.4.0", "hf"]
PARTICIPANTS = ("P01", "P03", "P04", "P06", "P07", "P22", "P28", "P30")
TEST_PARTICIPANTS = {"P06", "P22", "P30"}
NEGATIVE_PARTICIPANTS = {"P28", "P30"}
HUB_TEST_NUMBERS = {"41", "42", "09"}
OWN_TEST_CLIPS = {"basic_fold_1", "fold_stack_unstack_unfold_cloths_1"}
WINDOW_S = 60
MIN_WASH_S = 10.0
MIN_COOK_S = 5.0
MIN_GYRO_STD_RAD_S = 0.015
MIN_VISUAL_CHANGE = 1.0  # Mean absolute grayscale difference, 8-bit pixels.
MAX_SENSOR_GAP_S = 0.1
GYRO_DEGREES_WARNING_RAD_S = 20.0
THREADS = 2
FIELDS = [
    "clip_id", "source", "participant", "video_path", "sensors_path",
    "task_expected", "goal_met_expected", "split", "attack", "license",
    "label_status", "notes",
]
PLAN = ROOT / "tools/clip_sources.json"
MANIFEST = ROOT / "data/manifest_v2.csv"


def read_url(url: str) -> bytes:
    with urlopen(url, timeout=90) as response:
        return response.read()


def hf_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(filename, safe='/')}"


def write_new(path: Path, content: bytes) -> None:
    """Identical existing files are readable caches; never overwrite anything."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to change existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def seconds(timestamp: str) -> float:
    hours, minutes, sec = map(float, timestamp.split(":"))
    return hours * 3600 + minutes * 60 + sec


def coverage(intervals: list[tuple[float, float]], start: int) -> float:
    end = start + WINDOW_S
    pieces = sorted((max(a, start), min(b, end)) for a, b in intervals if a < end and b > start)
    total, right = 0.0, float(start)
    for a, b in pieces:
        total += max(0.0, b - max(a, right))
        right = max(right, b)
    return total


def select_epic(annotations: pd.DataFrame, files: set[str]) -> list[dict]:
    """Two nonoverlapping windows per person; annotations never imply completion."""
    jobs = []
    for participant in PARTICIPANTS:
        negative = participant in NEGATIVE_PARTICIPANTS
        candidates = []
        rows = annotations[annotations.participant_id == participant]
        for video, group in rows.groupby("video_id", sort=True):
            base = f"{participant}/meta_data/{video}"
            video_file = f"{participant}/videos/{video}.MP4"
            if not {base + "-gyro.csv", base + "-accl.csv", video_file} <= files:
                continue
            actions = [(seconds(r.start_timestamp), seconds(r.stop_timestamp),
                        str(r.verb), str(r.narration), str(r.narration_id))
                       for r in group.itertuples()]
            wash = [(a, b) for a, b, _, text, _ in actions if "wash" in text.lower()]
            cook = [(a, b) for a, b, verb, _, _ in actions if verb in {"cut", "cook", "chop", "slice", "stir", "fry"}]
            target = cook if negative else wash
            # Only use complete windows inside the annotated video extent.
            last_start = math.floor(max(b for _, b, _, _, _ in actions)) - WINDOW_S
            starts = {max(0, int(t) + delta) for a, b in target for t in (a, b)
                      for delta in (-60, -30, 0)}
            for start in sorted(starts):
                if start > last_start:
                    continue
                wash_s, cook_s = coverage(wash, start), coverage(cook, start)
                if (negative and (wash_s > 0 or cook_s < MIN_COOK_S)) or (not negative and wash_s < MIN_WASH_S):
                    continue
                ids = [identifier for a, b, _, text, identifier in actions
                       if a < start + WINDOW_S and b > start
                       and (negative or "wash" in text.lower())]
                candidates.append({"video_id": video, "start": start, "end": start + WINDOW_S,
                                   "wash_s": round(wash_s, 3), "annotation_ids": ids,
                                   "density": cook_s if negative else wash_s})
        chosen = []
        for item in sorted(candidates, key=lambda v: (-v["density"], v["video_id"], v["start"])):
            if any(item["video_id"] == other["video_id"] and
                   item["start"] < other["end"] and item["end"] > other["start"] for other in chosen):
                continue
            chosen.append(item)
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise ValueError(f"Cannot select two annotated sensor-backed windows for {participant}")
        for item in chosen:
            item.pop("density")
            item.update(source="epic", repo=EPIC_REPO, revision=EPIC_REV,
                        participant=participant, split="test" if participant in TEST_PARTICIPANTS else "train",
                        task="cooking_no_wash" if negative else "washing",
                        license="CC-BY-NC-4.0", goal="unknown")
            item["clip_id"] = f"{item['video_id']}_{item['start']:03d}_{item['end']:03d}"
            item["remote_video"] = f"{participant}/videos/{item['video_id']}.MP4"
            item["video_path"] = f"data/clips/epic/{item['clip_id']}.mp4"
            item["sensors_path"] = f"data/clips/epic/{item['clip_id']}_sensors.csv"
            jobs.append(item)
    return jobs


def hub_jobs(files: set[str]) -> list[dict]:
    numbers = {"01", "02", "41", "42", "13", "09", "03"}
    jobs = []
    for filename in sorted(files):
        if filename[:1] != "#" or filename[1:3] not in numbers or not filename.endswith(".mp4"):
            continue
        stem = Path(filename).stem
        task = "folding" if "Fold_Clothes" in stem else "tea" if "Make_Tea" in stem else "wiping" if "Wipe_" in stem else "tidying"
        jobs.append(dict(source="hub", repo=HUB_REPO, revision=HUB_REV,
                         clip_id="hub_" + stem.lstrip("#"), remote_video=filename,
                         video_path="data/clips/hub/" + stem.lstrip("#") + ".mp4",
                         sensors_path="", participant="hub_identity_unknown",
                         split="test" if filename[1:3] in HUB_TEST_NUMBERS else "train",
                         task=task, goal="unknown", license="hub-egocentric-eval"))
    if len(jobs) != 7:
        raise ValueError("Hub pinned repository no longer contains the seven expected files")
    return jobs


def existing_jobs() -> list[dict]:
    """'own' is the user's existing local collection, not a claim of authorship."""
    jobs = []
    for path in sorted((ROOT / "outputs/public_samples").rglob("*.mp4")):
        if path.parent.name not in {"egodex", "epic_kitchens"}:
            continue
        imu = path.with_name(path.stem + "_sensors.csv")
        jobs.append(dict(source="own", clip_id=path.stem, video_path=str(path.relative_to(ROOT)),
                         sensors_path=str(imu.relative_to(ROOT)) if imu.exists() else "",
                         participant="P01" if path.parent.name == "epic_kitchens" else "egodex_identity_unknown",
                         split="test" if path.stem in OWN_TEST_CLIPS else "train",
                         task="kitchen_unspecified" if path.parent.name == "epic_kitchens" else "folding",
                         goal="unknown", license="CC-BY-NC-4.0" if path.parent.name == "epic_kitchens" else "CC-BY-NC-ND",
                         original_source=path.parent.name))
    original = next((job for job in jobs if job["clip_id"] == "basic_fold_0"), None)
    if original is not None:
        jobs.append({**original, "clip_id": "basic_fold_0_controlled_reupload",
                     "attack": "controlled_reupload", "duplicate_of": "basic_fold_0"})
    return jobs


def manifest_bytes(jobs: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for job in jobs:
        notes = ["planned; inspect local paths for availability", "completion not established by task annotation"]
        if job["source"] == "epic":
            notes += [f"video={job['video_id']}; window=[{job['start']},{job['end']}); wash_s={job['wash_s']}",
                      f"annotations={ANNOTATION_REV}; repo={job['revision']}",
                      "derived excerpt; source axes preserved; EIS and alignment unvalidated"]
        elif job["source"] == "hub":
            notes += ["EVALUATION ONLY; no redistribution including derivatives; not training-eligible",
                      "user-requested whole-clip split; actor unknown; participant independence not verified",
                      "repo=" + job["revision"]]
        else:
            notes += ["existing local fixture; own denotes existing collection, not authorship",
                      "original_source=" + job["original_source"]]
            if job["original_source"] == "egodex":
                notes += ["user-requested clip-level split; actor unknown; previously used development footage"]
            if job.get("duplicate_of"):
                notes += ["controlled alias of " + job["duplicate_of"] + "; same source file; not an independent recording"]
        writer.writerow(dict(clip_id=job["clip_id"], source=job["source"], participant=job["participant"],
                             video_path=job["video_path"], sensors_path=job["sensors_path"],
                             task_expected=job["task"], goal_met_expected=job["goal"], split=job["split"],
                             attack=job.get("attack", "none"), license=job["license"], label_status="unverified", notes="; ".join(notes)))
    return stream.getvalue().encode()


def prepare() -> None:
    """Only metadata downloads. No dataset videos or parquet are fetched here."""
    raw = ROOT / "data/raw/catalogs"
    epic_bytes = read_url(f"https://huggingface.co/api/datasets/{EPIC_REPO}/revision/{EPIC_REV}")
    hub_bytes = read_url(f"https://huggingface.co/api/datasets/{HUB_REPO}/revision/{HUB_REV}")
    annotation_bytes = read_url(ANNOTATION_URL)
    license_bytes = read_url(hf_url(HUB_REPO, HUB_REV, "LICENSE"))
    print(license_bytes.decode(), flush=True)
    jobs = select_epic(pd.read_csv(io.BytesIO(annotation_bytes)),
                       {f["rfilename"] for f in json.loads(epic_bytes)["siblings"]})
    jobs += hub_jobs({f["rfilename"] for f in json.loads(hub_bytes)["siblings"]})
    jobs += existing_jobs()
    content = {"version": 2, "annotation_revision": ANNOTATION_REV,
               "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(), "jobs": jobs,
               "blocked": {"ha": "401/not found: list authenticated repo and inspect parquet before selection/conversion"}}
    # Validate all outputs before creating any data file.
    for path, data in [(PLAN, json.dumps(content, indent=2).encode() + b"\n"), (MANIFEST, manifest_bytes(jobs))]:
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace {path}; use a new versioned manifest/plan")
    for name, data in [("EPIC_100_train.csv", annotation_bytes), ("hub-LICENSE.txt", license_bytes)]:
        write_new(raw / name, data)
    write_new(PLAN, json.dumps(content, indent=2).encode() + b"\n")
    write_new(MANIFEST, manifest_bytes(jobs))
    status()


def apply_requested_split() -> None:
    """Explicit manifest-only update authorized by the user; never changes media."""
    plan = json.loads(PLAN.read_text())
    previous_manifest = MANIFEST.read_bytes()
    # Guard manual labels/edits: only migrate a manifest matching its current plan.
    # Version 1 used different explanatory notes; compare the stable data fields.
    old_rows = list(csv.DictReader(io.StringIO(previous_manifest.decode())))
    expected_ids = {job["clip_id"] for job in plan["jobs"]}
    if {row["clip_id"] for row in old_rows} != expected_ids:
        raise ValueError("Manifest has manual rows not represented by the plan; refusing migration")
    for row in old_rows:
        job = next(job for job in plan["jobs"] if job["clip_id"] == row["clip_id"])
        if any(row[key] != job[key] for key in ("source", "participant", "video_path", "sensors_path", "split")):
            raise ValueError("Manifest differs from the pinned plan; refusing migration")
        if row["label_status"] != "unverified" or row["goal_met_expected"] != job["goal"]:
            raise ValueError("Manifest contains reviewed labels; refusing migration")
    jobs = [job for job in plan["jobs"] if job["source"] == "epic"]
    jobs += hub_jobs({job["remote_video"] for job in plan["jobs"] if job["source"] == "hub"})
    jobs += existing_jobs()
    if sum(job["source"] == "own" for job in jobs) != 6:
        raise ValueError("Expected five existing videos plus the controlled reupload alias")
    new_plan = {**plan, "version": 2, "jobs": jobs,
                "blocked": {"ha": plan["blocked"]["ha"]}}
    backup = ROOT / "outputs/corpus_backups" / digest(MANIFEST)
    write_new(backup / "manifest.csv", previous_manifest)
    write_new(backup / "clip_sources.json", PLAN.read_bytes())
    # This command is the sole explicit exception to immutable output creation.
    # The user authorized correcting data/manifest.csv, not any raw/media files.
    manifest_content = manifest_bytes(jobs)
    plan_content = json.dumps(new_plan, indent=2).encode() + b"\n"
    if MANIFEST.read_bytes() != manifest_content:
        MANIFEST.write_bytes(manifest_content)
    if PLAN.read_bytes() != plan_content:
        PLAN.write_bytes(plan_content)
    status()


def normalize_epic(gyro_path: Path, accel_path: Path, start: float, end: float) -> pd.DataFrame:
    gyro, accel = pd.read_csv(gyro_path), pd.read_csv(accel_path)
    for frame, columns in [(gyro, ["GyroX", "GyroY", "GyroZ"]), (accel, ["AcclX", "AcclY", "AcclZ"])]:
        values = frame[["Milliseconds"] + columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(values) < 2 or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError("Sensor timestamps/values must be finite and strictly increasing")
    gt, at = gyro.Milliseconds.to_numpy() / 1000, accel.Milliseconds.to_numpy() / 1000
    mask = (gt >= start) & (gt < end) & (gt >= at[0]) & (gt <= at[-1])
    times = gt[mask]
    if len(times) < 2 or times[0] - start > MAX_SENSOR_GAP_S or end - times[-1] > MAX_SENSOR_GAP_S:
        raise ValueError("Sensors do not cover the requested video window")
    if np.max(np.diff(times)) > MAX_SENSOR_GAP_S:
        raise ValueError("Gyroscope contains a gap exceeding 100 ms")
    indices = np.searchsorted(at, times, side="right").clip(1, len(at) - 1)
    if np.any(at[indices] - at[indices - 1] > MAX_SENSOR_GAP_S):
        raise ValueError("Acceleration interpolation would cross a gap exceeding 100 ms")
    xyz = gyro.loc[mask, ["GyroX", "GyroY", "GyroZ"]].to_numpy()
    if np.quantile(np.abs(xyz), 0.99) > GYRO_DEGREES_WARNING_RAD_S:
        warnings.warn("Gyro values look like deg/s or extreme motion; source SI assumption needs checking", stacklevel=2)
    output = pd.DataFrame({"t": times - start, "gx": xyz[:, 0], "gy": xyz[:, 1], "gz": xyz[:, 2]})
    for target, source in zip(("ax", "ay", "az"), ("AcclX", "AcclY", "AcclZ")):
        output[target] = np.interp(times, at, accel[source])
    return output


def hf_download(repo: str, revision: str, filename: str, directory: Path) -> Path:
    subprocess.run(HF_COMMAND + ["download", repo, filename, "--repo-type", "dataset",
                               "--revision", revision, "--local-dir", str(directory)], check=True)
    return directory / filename


def probe(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height:format=duration", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def visual_change(path: Path) -> float:
    result = subprocess.run(["ffmpeg", "-v", "error", "-threads", str(THREADS), "-i", str(path),
                             "-vf", "fps=1,scale=160:90,format=gray", "-threads", str(THREADS),
                             "-f", "rawvideo", "-"], check=True, capture_output=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 90, 160).astype(float)
    return float(np.median(np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2)))) if len(frames) > 1 else 0.0


def cached(job: dict) -> bool:
    destination = ROOT / job["video_path"]
    paths = [destination, destination.with_suffix(".provenance.json")]
    if job["sensors_path"]:
        paths.append(ROOT / job["sensors_path"])
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths):
        raise FileExistsError(f"Partial/unowned existing output; refusing overwrite: {destination}")
    record = json.loads(paths[1].read_text())
    if record["job"] != job or record["video_sha256"] != digest(destination):
        raise ValueError(f"Existing video/provenance mismatch: {destination}")
    if job["sensors_path"] and record["sensors_sha256"] != digest(paths[2]):
        raise ValueError(f"Existing sensor/provenance mismatch: {paths[2]}")
    print("CACHED", job["clip_id"], flush=True)
    return True


def publish(job: dict, video: Path, sensor: pd.DataFrame | None, details: dict) -> None:
    destination = ROOT / job["video_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Hard link publishes without overwriting or duplicating the staged video.
    os.link(video, destination)
    record = {"job": job, "video_sha256": digest(video), **details}
    if sensor is not None:
        sensor_path = ROOT / job["sensors_path"]
        write_new(sensor_path, sensor.to_csv(index=False, float_format="%.10g").encode())
        record["sensors_sha256"] = digest(sensor_path)
    write_new(destination.with_suffix(".provenance.json"), json.dumps(record, indent=2).encode() + b"\n")
    print("READY", job["clip_id"], flush=True)


def fetch_epic_group(jobs: list[dict]) -> list[str]:
    pending = [job for job in jobs if not cached(job)]
    if not pending:
        return []
    root = ROOT / "data/raw/epic"
    root.mkdir(parents=True, exist_ok=True)
    # This brand-new directory establishes ownership. Never delete files from older runs.
    owned = Path(tempfile.mkdtemp(prefix=jobs[0]["video_id"] + "-", dir=root))
    base = f"{jobs[0]['participant']}/meta_data/{jobs[0]['video_id']}"
    gyro = hf_download(EPIC_REPO, EPIC_REV, base + "-gyro.csv", owned)
    accel = hf_download(EPIC_REPO, EPIC_REV, base + "-accl.csv", owned)
    sensors = {}
    skipped = []
    for job in pending:
        sensor = normalize_epic(gyro, accel, job["start"], job["end"])
        motion = float(np.linalg.norm(sensor[["gx", "gy", "gz"]].std(ddof=0)))
        if motion < MIN_GYRO_STD_RAD_S:
            reason = f"Mostly still: gyro std {motion:.4f} rad/s; revise plan explicitly"
            print("SKIPPED", job["clip_id"], reason, flush=True)
            write_new(owned / (job["clip_id"] + ".skipped.json"), json.dumps({"job": job, "reason": reason}).encode())
            skipped.append(job["clip_id"])
            continue
        sensors[job["clip_id"]] = sensor
    pending = [job for job in pending if job["clip_id"] in sensors]
    if not pending:
        return skipped
    source = hf_download(EPIC_REPO, EPIC_REV, jobs[0]["remote_video"], owned)
    source_hash = digest(source)
    for job in pending:
        cut = owned / (job["clip_id"] + ".mp4")
        command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", str(THREADS),
                   "-ss", str(job["start"]), "-i", str(source), "-t", str(WINDOW_S), "-map", "0:v:0",
                   "-an", "-c:v", "libx264", "-threads", str(THREADS), "-preset", "veryfast",
                   "-crf", "23", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-movflags", "+faststart", str(cut)]
        subprocess.run(command, check=True)
        metadata = probe(cut)
        if abs(float(metadata["format"]["duration"]) - WINDOW_S) > 0.1:
            raise ValueError("Cut is shorter/longer than 60 s; refusing to publish")
        change = visual_change(cut)
        if change < MIN_VISUAL_CHANGE:
            reason = f"Mostly still images: change={change:.4f}; revise plan explicitly"
            print("SKIPPED", job["clip_id"], reason, flush=True)
            write_new(owned / (job["clip_id"] + ".skipped.json"), json.dumps({"job": job, "reason": reason}).encode())
            skipped.append(job["clip_id"])
            cut.unlink()
            continue
        publish(job, cut, sensors[job["clip_id"]],
                {"source_sha256": source_hash, "gyro_sha256": digest(gyro), "accel_sha256": digest(accel),
                 "ffmpeg_command": command, "probe": metadata, "visual_change": change,
                 "sensor_units": "s,rad/s,m/s^2", "axes": "published GyroX/Y/Z and AcclX/Y/Z unchanged",
                 "calibration": "not calibrated; no time offset or stabilization correction"})
        cut.unlink()  # Only our newly created staged link; published video remains.
    source.unlink()  # Explicitly requested: delete only THIS invocation's full-length source.
    print("REMOVED NEW SOURCE", str(source.relative_to(ROOT)), flush=True)
    return skipped


def inspect_ha() -> int:
    """Authenticated listing, with no assumptions about unknown episode layout."""
    result = subprocess.run(HF_COMMAND + ["datasets", "ls", HA_REPO, "--recursive", "--tree"], check=False)
    if result.returncode:
        print("HA BLOCKED: confirm repo ID/access and run `uvx --from huggingface_hub hf auth login`.")
    else:
        print("STOP: inspect this layout, select head videos/parquets, then print parquet schema before mapping units/axes.")
    return result.returncode


def status() -> None:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    print("\nMANIFEST counts (planned + existing; not a download-success count):")
    for fields in [("source",), ("task_expected",), ("split",), ("source", "task_expected", "split")]:
        counts = Counter(tuple(row[key] for key in fields) for row in rows)
        print(" / ".join(fields), dict(sorted(counts.items())))
    ready = [row["clip_id"] for row in rows if (ROOT / row["video_path"]).is_file()]
    distinct = {row["video_path"] for row in rows if (ROOT / row["video_path"]).is_file()}
    print("Video entries present:", len(ready), "/", len(rows), f"({len(distinct)} distinct files)", ready)
    print("No associated sensors:", [row["clip_id"] for row in rows if not row["sensors_path"]])
    print("Sensors planned but not downloaded:", [row["clip_id"] for row in rows
          if row["sensors_path"] and not (ROOT / row["sensors_path"]).is_file()])
    paths = {p for p in (ROOT / "data").rglob("*") if p.is_file()}
    paths |= {ROOT / row[key] for row in rows for key in ("video_path", "sensors_path") if row[key] and (ROOT / row[key]).is_file()}
    size = sum(p.stat().st_size for p in paths)
    print(f"Corpus logical disk size (data + referenced existing clips/CSVs, no HF tool cache): {size / 1024**2:.2f} MiB")
    print("Blocked: HA access/schema. Own includes one controlled alias, not six independent videos. Hub is evaluation-only.")
    print("Full currently-resolved download command:\n  uv run python tools/fetch_clips.py fetch --source all --accept-hub-evaluation-license")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "fetch", "status", "inspect-ha", "apply-requested-split"])
    parser.add_argument("--source", choices=["epic", "hub", "all"], default="all")
    parser.add_argument("--clip-id", help="Fetch only this planned clip (use with --source epic for smoke test)")
    parser.add_argument("--limit", type=int, help="Maximum clips across selected sources (1 for smoke test)")
    parser.add_argument("--accept-hub-evaluation-license", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "prepare":
        prepare()
    elif args.command == "status":
        status()
    elif args.command == "inspect-ha":
        return inspect_ha()
    elif args.command == "apply-requested-split":
        apply_requested_split()
    else:
        plan = json.loads(PLAN.read_text())
        jobs = [job for job in plan["jobs"] if job["source"] in ({"epic", "hub"} if args.source == "all" else {args.source})]
        if args.clip_id:
            jobs = [job for job in jobs if job["clip_id"] == args.clip_id]
            if not jobs:
                parser.error("--clip-id does not match a planned clip in the selected source")
        if args.limit:
            jobs = jobs[:args.limit]
        if any(j["source"] == "hub" for j in jobs):
            print((ROOT / "data/raw/catalogs/hub-LICENSE.txt").read_text(), flush=True)
            if not args.accept_hub_evaluation_license:
                parser.error("Read the Hub license; pass --accept-hub-evaluation-license for local evaluation only")
        groups = defaultdict(list)
        for job in jobs:
            if job["source"] == "epic":
                groups[job["video_id"]].append(job)
        skipped = []
        for group in groups.values():
            skipped.extend(fetch_epic_group(group))
        for job in jobs:
            if job["source"] != "hub" or cached(job):
                continue
            raw = ROOT / "data/raw/hub"
            raw.mkdir(parents=True, exist_ok=True)
            owned = Path(tempfile.mkdtemp(prefix="download-", dir=raw))
            video = hf_download(job["repo"], job["revision"], job["remote_video"], owned)
            metadata = probe(video)
            publish(job, video, None, {"probe": metadata, "usage": "evaluation only; no redistribution"})
            video.unlink()  # Newly downloaded hardlink now lives under clips/.
        status()
        if skipped:
            print("INCOMPLETE: mostly still windows skipped:", skipped)
            return 2
        if args.source == "all":
            print("INCOMPLETE: HA access/schema remains blocked; no HA downloads attempted.")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
