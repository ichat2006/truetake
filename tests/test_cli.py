import json
from pathlib import Path

import numpy as np
import pytest

from truetake.cli import main
from truetake.models import Recording


@pytest.fixture
def loaded_clip(monkeypatch):
    rec = Recording(
        "example", "example.mp4", np.array([0.0, 0.1]),
        {"width": 320, "height": 180, "display_width": 320,
         "display_height": 180, "fps": 10.0, "rotation": 0,
         "duration_s": 0.2, "duration_estimated": False},
        None,
    )
    monkeypatch.setattr("truetake.cli.load_recording", lambda *args: rec)
    return rec


def test_summary_does_not_claim_quality_or_verification(loaded_clip, capsys):
    assert main(["example.mp4"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "ingest"
    assert summary["review_status"] == "not_run"
    assert summary["sensor_status"] == "no motion data"
    assert summary["sensor_rate_hz"] is None
    assert summary["sensor_samples"] == 0


def test_contract_is_loaded_not_evaluated(loaded_clip, tmp_path, capsys):
    contract = tmp_path / "fold.yaml"
    contract.write_text("task: fold a T-shirt\n")
    assert main(["example.mp4", "--contract", str(contract)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["task"] == "fold a T-shirt"
    assert summary["review_status"] == "not_run"


@pytest.mark.parametrize("content", ["", "[]", "task: 5", "task: ''", "task: [", "task: fold\ndraft: true\n"])
def test_invalid_contract_has_nonzero_exit(tmp_path, content, capsys):
    contract = tmp_path / "fold.yaml"
    contract.write_text(content)
    with pytest.raises(SystemExit) as error:
        main(["example.mp4", "--contract", str(contract)])
    assert error.value.code == 2
    assert "truetake:" in capsys.readouterr().err


def test_missing_input_has_friendly_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.mp4")])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert "Video file not found" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("suffix", ["data", "data/generated"])
def test_basic_output_cannot_write_under_data(loaded_clip, suffix, capsys):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(SystemExit) as error:
        main(["example.mp4", "--basic", "--output-dir", str(root / suffix)])
    assert error.value.code == 2
    assert "outside data/" in capsys.readouterr().err


def test_profile_rotation_must_match_recording(loaded_clip, tmp_path, capsys):
    profile = {"schema_version": 1, "status": "provisional", "min_match_score": None,
               "source_rotation_deg": 90, "axis_mapping": {
                   "flow_pan": {"axis": "gyro_gx", "sign": 1},
                   "flow_tilt": {"axis": "gyro_gy", "sign": 1},
                   "flow_roll": {"axis": "gyro_gz", "sign": 1}}}
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(SystemExit) as error:
        main(["example.mp4", "--gyro-profile", str(path)])
    assert error.value.code == 2
    assert "rotation differs" in capsys.readouterr().err
