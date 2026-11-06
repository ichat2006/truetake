import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from truetake.calibration import load_profile
from truetake.contracts import load_contract


ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_washing_contract_preserves_folding_requirement_policy():
    fold = load_contract(str(ROOT / "contracts/mercor_folding.yaml"))
    wash = load_contract(str(ROOT / "contracts/wash_dishes.yaml"))
    assert wash["task"] == "wash dishes"
    assert wash["goal"]["description"] == "Dishes rinsed and set aside."
    assert wash["requirements"] == fold["requirements"]
    assert wash["duration"] == fold["duration"]
    assert any("sink" in rule.get("description", "") for rule in wash["process"])
    assert next(rule for rule in wash["process"] if rule["id"] == "idle")["max_idle_s"] == 3.


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_variants_are_real_encoded_files_and_sensor_copies_preserve_inputs(tmp_path, monkeypatch):
    variants = tool("make_demo_variants")
    (tmp_path / "outputs").mkdir()
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=5:duration=60",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(source)], check=True, capture_output=True)
    own_imu = tmp_path / "source_sensors.csv"
    other_imu = tmp_path / "other_sensors.csv"
    own_imu.write_text("t,gx,gy,gz,ax,ay,az\n0,1,2,3,0,0,9.81\n")
    other_imu.write_text("t,gx,gy,gz,ax,ay,az\n0,3,2,1,0,0,9.81\n")
    names = {spec[key] for spec in variants.VARIANTS for key in ("parent", "other") if key in spec}
    rows = [dict(clip_id=name, source="own", participant="fixture", video_path="source.mp4",
                 sensors_path="other_sensors.csv" if name == "P22_109_1851_1911" else "source_sensors.csv",
                 task_expected="folding", goal_met_expected="unknown", split="test" if name == "basic_fold_1" else "train",
                 attack="none", license="fixture", label_status="unverified", notes="") for name in sorted(names)]
    manifest = tmp_path / "data/manifest.csv"
    manifest.parent.mkdir()
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, source, own_imu, other_imu)}
    monkeypatch.setattr(variants, "ROOT", tmp_path)
    monkeypatch.setattr(variants, "DESTINATION", tmp_path / "data/clips/variants")
    made = variants.create_variants()
    assert len(made) == 5
    for row in made:
        assert row["source"] == "variant" and row["label_status"] == "verified"
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,pix_fmt:format=duration", "-of", "json", str(tmp_path / row["video_path"])],
                               check=True, capture_output=True, text=True)
        result = json.loads(probe.stdout)
        assert result["streams"][0]["codec_name"] == "h264"
        assert result["streams"][0]["pix_fmt"] == "yuv420p"
        assert float(result["format"]["duration"]) == pytest.approx(60., abs=.3)
        if row["defect_type"] in {"blur_mid", "dark"}:
            assert row["sensors_path"] == ""
        else:
            expected = other_imu if row["defect_type"] == "swap" else own_imu
            assert (tmp_path / row["sensors_path"]).read_bytes() == expected.read_bytes()
    assert made[0]["split"] == "test"
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in before.items())
    with pytest.raises(FileExistsError):
        variants.create_variants()
