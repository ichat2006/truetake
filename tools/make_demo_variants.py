"""Create the five operator-specified fixtures without replacing existing data."""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/clips/variants"
THREADS = 2
CRF = 18
TIMEOUT_S = 600
VARIANTS = [
    {"clip_id": "basic_fold_1_blur_mid", "parent": "basic_fold_1", "defect": "blur_mid",
     "filter": "gblur=sigma=20:enable='between(t,10,18)'", "expected": "approved with 10–18 s excluded"},
    {"clip_id": "fold_stack_unstack_unfold_cloths_0_dark", "parent": "fold_stack_unstack_unfold_cloths_0",
     "defect": "dark", "filter": "eq=brightness=-0.45", "expected": "rejected: too dark"},
    {"clip_id": "splice_P03_109_P04_104", "parent": "P03_109_685_745", "other": "P04_104_592_652",
     "defect": "splice", "expected": "sensor mismatch in second half"},
    {"clip_id": "P06_102_575_635_slow2x", "parent": "P06_102_575_635", "defect": "slow2x",
     "filter": "setpts=2.0*PTS", "expected": "sensor mismatch"},
    {"clip_id": "P07_102_059_119_swap", "parent": "P07_102_059_119", "other": "P22_109_1851_1911",
     "defect": "swap", "expected": "sensor mismatch"},
]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def command(spec: dict, parents: dict, output: Path) -> list[str]:
    source = ROOT / parents[spec["parent"]]["video_path"]
    args = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", str(THREADS), "-i", str(source)]
    if spec["defect"] == "splice":
        args += ["-threads", str(THREADS), "-i", str(ROOT / parents[spec["other"]]["video_path"]),
                 "-filter_complex_threads", "1", "-filter_complex",
                 "[0:v:0]trim=start=0:end=30,setpts=PTS-STARTPTS[a];"
                 "[1:v:0]trim=start=0:end=30,setpts=PTS-STARTPTS[b];"
                 "[a][b]concat=n=2:v=1:a=0[v]", "-map", "[v]"]
    else:
        args += ["-filter_threads", "1", "-vf", spec["filter"], "-map", "0:v:0"]
    if spec["defect"] == "slow2x":
        args += ["-t", "60"]
    return args + ["-an", "-c:v", "libx264", "-threads", str(THREADS), "-preset", "veryfast",
                   "-crf", str(CRF), "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                   "-movflags", "+faststart", str(output)]


def create_variants() -> list[dict]:
    with (ROOT / "data/manifest.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        parents = {row["clip_id"]: row for row in reader}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = DESTINATION / "manifest.csv"
    if manifest.exists():
        raise FileExistsError("Variant manifest already exists; existing data will not be modified")
    rows, skipped = [], []
    for spec in VARIANTS:
        parent = parents[spec["parent"]]
        video = DESTINATION / (spec["clip_id"] + ".mp4")
        sensor = DESTINATION / (spec["clip_id"] + "_sensors.csv")
        provenance = DESTINATION / (spec["clip_id"] + ".json")
        if any(path.exists() for path in (video, sensor, provenance)):
            raise FileExistsError(f"Variant already exists: {spec['clip_id']}")
        required = [ROOT / parent["video_path"]]
        sensor_source = None
        if spec["defect"] in {"splice", "slow2x", "swap"}:
            sensor_parent = parents[spec["other"]] if spec["defect"] == "swap" else parent
            sensor_source = ROOT / sensor_parent["sensors_path"]
            required.append(sensor_source)
        if spec["defect"] == "splice":
            required.append(ROOT / parents[spec["other"]]["video_path"])
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            skipped.append({"clip_id": spec["clip_id"], "missing": missing})
            print("SKIPPED", spec["clip_id"], missing, flush=True)
            continue
        # Only publish complete new files. Staging lives outside immutable data/.
        with tempfile.TemporaryDirectory(prefix="truetake-variant-", dir=ROOT / "outputs") as temporary:
            stage = Path(temporary) / video.name
            encode = None
            if spec["defect"] == "swap":
                shutil.copyfile(required[0], stage)
            else:
                encode = command(spec, parents, stage)
                subprocess.run(encode, check=True, timeout=TIMEOUT_S)
            os.link(stage, video)
        if sensor_source is not None:
            with sensor_source.open("rb") as source, sensor.open("xb") as target:
                shutil.copyfileobj(source, target)
        details = {**spec, "command": encode, "video_sha256": sha256(video),
                   "sensor_source": str(sensor_source) if sensor_source else None,
                   "sensor_sha256": sha256(sensor) if sensor_source else None,
                   "label_scope": "verified synthetic manipulation; expected pipeline outcome is not an observed verdict"}
        with provenance.open("x") as handle:
            json.dump(details, handle, indent=2)
        row = {**parent, "clip_id": spec["clip_id"], "source": "variant",
               "video_path": str(video.relative_to(ROOT)),
               "sensors_path": str(sensor.relative_to(ROOT)) if sensor_source else "",
               "attack": spec["defect"], "label_status": "verified", "parent_clip_id": spec["parent"],
               "defect_type": spec["defect"], "expected_outcome": spec["expected"],
               "task_expected": "folding" if spec["defect"] in {"blur_mid", "dark"} else "washing",
               "notes": "Synthetic demo fixture; " + details["label_scope"]}
        rows.append(row)
        print("CREATED", spec["clip_id"], "split=" + row["split"], flush=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields + ["parent_clip_id", "defect_type", "expected_outcome"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with manifest.open("x") as handle:
        handle.write(stream.getvalue())
    with (ROOT / "outputs/variant_creation.json").open("w") as handle:
        json.dump({"created": [row["clip_id"] for row in rows], "skipped": skipped,
                   "manifest": str(manifest)}, handle, indent=2)
    return rows


if __name__ == "__main__":
    create_variants()
