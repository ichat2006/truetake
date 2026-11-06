# TrueTake

For the hackathon batch demo, open [the demo guide](docs/DEMO.md). It includes
the report URL, per-video Rerun views, regeneration commands, and the remaining
AI/labeling requirements.

TrueTake reviews first-person task recordings collected for robot training.
The current implementation loads video timing and optional sensor data, builds
a shared per-frame table, and measures brightness, blur, image change, and
optical-flow camera motion.
It also measures diagnostic gyro/video alignment and clock offset. The batch
pipeline now produces contract-based quality cards, Claude task assessments,
approved-only value ranking, a clickable HTML report, and per-clip Rerun views.
Calibration remains deferred, so sensor verification is inconclusive.
An opt-in `--detector-mode onepass` shares one decode across local CPU detectors
and saves selected Claude images during that pass; see the demo guide for
the measured comparison and the original `multipass` fallback.
Saved reviews can now produce corrected videos, edit maps and retimed sensor
CSVs without repeating inference:

```bash
uv run python export.py outputs/demo/batch.json --output-dir outputs/demo/exports --rerun
```

Accepted copies and pending copies are stored separately. Originals stay intact;
the report links the derivatives and manifests. See the demo guide for timing,
privacy, audio and sensor-axis policies.
The lightweight single-clip CLI reports `review_status: "not_run"`; use the
batch entry point for complete review:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --env-file .env python batch.py outputs/public_samples --contract contracts/mercor_folding.yaml --output-dir outputs/demo --include-duplicate
```

This runtime command uses the existing standard `ANTHROPIC_API_KEY`; application
code never opens the environment file. Credentials and generated outputs are
ignored by Git. See [the active lab-input contract](contracts/mercor_folding.yaml)
and [the evidence-first review prompt](prompts/review_v2.md). Completion is required
by default, brief hand exits are allowed, and detected faces trigger blur.

## Setup

Use Python 3.11 inside WSL. The repository's `.python-version` selects it for uv.
FFmpeg and ffprobe must be installed and available on `PATH`.

```bash
uv sync --locked
uv run python cli.py --help
```

Dependencies are already declared in `pyproject.toml`; ask before adding any.
PyTorch and torchvision are configured to use the CPU package index.
The loader does not initialize models or make API calls, and does not read `.env`.

## Load a recording

```bash
uv run python cli.py data/clips/clip_001.mp4 --sensors data/clips/clip_001_sensors.csv --contract contracts/fold.yaml
```

The installed `uv run truetake` command accepts the same arguments.
The output is a JSON summary of frame count, duration, dimensions, rotation,
average FPS, and sensor sample count/rate. Inputs are not modified and this
command creates no output files.

## Basic frame measurements

```bash
uv run python cli.py outputs/public_samples/egodex/basic_fold_1.mp4 --contract contracts/fold.yaml --basic
uv run python cli.py outputs/public_samples/epic_kitchens/P01_102_000_060.mp4 --sensors outputs/public_samples/epic_kitchens/P01_102_000_060_sensors.csv --basic
```

`--basic` writes `outputs/<clip_id>_frames.json` and prints its path. Use
`--output-dir` to choose a different generated-output directory outside `data/`.
Each file is a JSON array of records with these columns:

| Columns | Meaning |
| --- | --- |
| `source_frame` | Original presentation-order frame index |
| `t`, `sample_end` | Actual selected-frame time and next selected time (last: clip end), seconds |
| `clip_meta`, `contract` | Clip metadata and supplied contract on row 0; null elsewhere |
| `sensor_present` | Whether a sensor file was supplied |
| `gx`, `gy`, `gz` | Interpolated angular velocity, rad/s |
| `ax`, `ay`, `az` | Interpolated acceleration, m/s² |
| `imu_valid` | Sensor interpolation is supported at this timestamp |
| `brightness` | Mean grayscale intensity, 0–255 |
| `blur_var` | Variance of the grayscale Laplacian at analysis resolution |
| `frame_change` | Mean absolute change from the preceding sampled image, divided by 255 |

The table selects the first available source frame at or after each 10 Hz
target, without duplicating frames. Variable-rate and low-rate sources may
produce uneven or fewer samples. `sample_end` describes the sampling interval;
it does not prove that every instant in that interval was observed.

Decoding proceeds sequentially through the entire clip. Sampled images are
rotated according to metadata and shrunk to at most 320 pixels wide. Every
sample is measured, including dark or frozen images; no Plan A gates exist.
Only the current and previous image are retained in memory.

Sensor columns are null outside coverage and across gaps longer than 0.1 s.
The `gx/gy/gz` columns contain instantaneous interpolation. Additional
`gyro_gx/gyro_gy/gyro_gz` columns contain centered 0.2-second means computed at
the original sensor rate before sampling for alignment. Smoothing does not
cross sensor gaps. The first `frame_change` is null because it has no
preceding image. Brightness and blur have no pass/fail thresholds yet; calibration
and task-window checks are separate future steps.

## Camera motion

```bash
uv run python cli.py outputs/public_samples/egodex/basic_fold_1.mp4 --contract contracts/fold.yaml --flow
uv run python cli.py outputs/public_samples/epic_kitchens/P01_102_000_060.mp4 --sensors outputs/public_samples/epic_kitchens/P01_102_000_060_sensors.csv --flow
```

`--flow` includes basic measurements and adds these columns to the same JSON:

| Column | Meaning |
| --- | --- |
| `flow_t` | Actual midpoint of the sampled frame pair, seconds |
| `flow_dt` | Actual pair interval, seconds |
| `flow_pan`, `flow_tilt` | Image-center translation rates in analysis pixels/s; right/down positive |
| `flow_roll` | Image rotation in rad/s; clockwise positive in image coordinates |
| `flow_points` | Number of successfully tracked, in-frame points |
| `flow_inlier_fraction` | Fraction supporting the fitted transform |
| `flow_status` | `ok`, `no_previous`, `frame_gap`, `insufficient_features`, `tracking_failed`, `insufficient_tracks`, `fit_failed`, or `low_inliers` |

The detector tracks up to 200 corners with Lucas–Kanade optical flow and fits
a similarity transform with RANSAC. At least 20 tracked points and 50% inliers
are required. Translation is measured at image center to avoid mixing pure
roll into pan/tilt. Intervals longer than 0.2 s are not bridged.

Unreliable rates are null, including the first row. A textured stationary
image can yield a valid zero rate; a featureless image yields unknown motion.
The decoder runs in a separate pass over all samples, consistent with Plan B.
Hands or cloth can still dominate a good fit: `ok` indicates a usable image
transform, not proof that it represents camera rotation. These signals are
correlated against gyro data with `--verify`; the device axis mapping remains
uncalibrated until a separate calibration recording establishes it.

## Gyro alignment

```bash
uv run python cli.py outputs/public_samples/epic_kitchens/P01_102_000_060.mp4 --sensors outputs/public_samples/epic_kitchens/P01_102_000_060_sensors.csv --verify
```

`--verify` includes `--flow` and writes a `gyro_alignment` object on row 0 of
the shared table. The CLI also prints that object and a `verification.gyro`
CheckResult. The measurement function reads only table signals; the check
reads only the stored measurement and table metadata.

The search compares video at `t` to gyro at `t + offset_s`, from −2 to +2 s in
0.1 s steps. Video rates are smoothed with a centered 0.2 s mean. Correlations
are mean-centered and scale-independent. Invalid samples, gaps, and times
outside the sensor recording are excluded, not extrapolated. At least 50
valid pairs spanning 5 s and two informative axes are required.

Without a calibration profile, the detector explores signed one-to-one axis
permutations and chooses a common lag. This search can inflate apparent
agreement; its score is diagnostic and never automatically passes a clip.
Per-axis peak offsets must agree within 0.2 s for a calibrated verdict.
Offsets over 1 s produce a warning flag. A best lag at the search boundary
remains inconclusive.

The optional API calibration profile is stored under row 0
`clip_meta.gyro_calibration`: `axis_mapping` maps each of `flow_pan`,
`flow_tilt`, and `flow_roll` to an `axis` (`gyro_gx`, `gyro_gy`, or `gyro_gz`)
and a `sign` (−1 or +1), using each sensor axis once. `min_match_score` must
come from training-clip calibration and be between 0 and 1. Tests exercise fixed
profiles; the downloaded real recordings remain uncalibrated. The CLI can now
export and load provisional mapping profiles, always with a null threshold.

Missing sensors return `passed: null` and reason `"no motion data"`. Flat
motion, inadequate overlap, inconsistent axis lags, and uncalibrated results
also remain inconclusive. Window diagnostics are available below; automatic
splice verdicts and a slow-motion stretch search are not implemented.

## Calibration diagnostics

```bash
uv run python cli.py outputs/public_samples/epic_kitchens/P01_102_000_060.mp4 --sensors outputs/public_samples/epic_kitchens/P01_102_000_060_sensors.csv --diagnose-gyro
```

This includes verification and writes three additional files under `outputs/`:

- `<clip_id>_alignment.png`: normalized video/gyro traces, correlation-versus-lag curves, and window results.
- `<clip_id>_alignment.json`: full diagnostic evidence and summary counts, also stored in row 0 `gyro_diagnostics` of the table.
- `<clip_id>_gyro_candidate.json`: a provisional mapping, source segment, and validation limitations. Omitted when no mapping can be measured.

The first 10 seconds estimate a mapping and reference offset. Later 5-second
windows advance by 1 second and hold that mapping fixed. Each window reports
both correlation at the original reference offset and the best local offset.
A separate exploratory fit measures mapping agreement; it does not replace
the fixed mapping. Whole-clip lag curves also hold the first-segment mapping
fixed. Plotted traces use that first segment's reference offset.

Windows require at least 40 valid pairs spanning 3.9 seconds, allowing some
missing frames in a five-second interval. Gyro context outside the video window
is retained for lag testing; video outside the window is excluded before
smoothing. Invalid samples and long gaps remain missing. Missing sensors
produce an explanatory plot and no candidate. Overlapping windows from one
recording are not independent validation recordings.

To reuse a candidate for diagnostics:

```bash
uv run python cli.py outputs/public_samples/epic_kitchens/P01_102_000_060.mp4 --sensors outputs/public_samples/epic_kitchens/P01_102_000_060_sensors.csv --gyro-profile outputs/P01_102_000_060_gyro_candidate.json
```

The loader accepts only version-1 provisional profiles with no acceptance
threshold, checks the signed axis permutation, and rejects a different display
rotation. Device identity remains unconfirmed: do not assume the mapping
transfers to another camera/setup. Reusing a candidate fixes the mapping but
still estimates offset per recording and returns an inconclusive verification
result. It never promotes a candidate to a calibrated verifier.

On the downloaded EPIC excerpt, the first-segment candidate was unstable:
reference −0.6 s; 45/46 later windows measurable; 0/45 matching the full axis
mapping; 12/45 within 0.2 s of the reference offset. Median fixed-reference
correlation was approximately 0.14. These results do not establish an authentic
or tampered verdict, or identify the cause of disagreement. A known camera-axis
calibration recording and validation across multiple genuine recordings remain
necessary before choosing acceptance thresholds.

`contracts/fold.yaml` retains the original design's strict example for legacy
checks. The batch defaults to `contracts/mercor_folding.yaml`, which reflects
the operator-approved collection policy. Contract loading validates the schema;
the single-clip diagnostic CLI does not evaluate quality rules.

For video without motion data:

```bash
uv run python cli.py path/to/video.mp4 --contract contracts/fold.yaml
```

This preserves `Recording.imu = None` and reports `sensor_status: "no motion data"`.
A supplied but missing or invalid sensor file is an error, not a video-only run.

## Sensor input contract

The current loader accepts an already-normalized CSV with these columns:

| Column | Units |
| --- | --- |
| `t` | Float seconds relative to video start |
| `gx`, `gy`, `gz` | Angular velocity in rad/s, in documented device axes |
| `ax`, `ay`, `az` | Acceleration in m/s², in the same device axes |

All seven columns must contain finite numbers. Timestamps must increase
strictly; duplicate or out-of-order timestamps are rejected. Extra columns
are ignored. A negative `t` is allowed for samples recorded before video start.

The loader does not guess units, subtract the first sensor timestamp, or remap
axes. A phone-export adapter will follow once the app's format, units, and
video/sensor clock relationship are known. Do not overwrite raw recordings to
normalize them; use an output directory outside `data/` for converted files.

## Video timing

Frame times come from ffprobe presentation timestamps, including variable-rate
video; they are normalized so the first video frame is at 0 seconds. They are
not calculated from nominal FPS. `Recording.frame_times` contains one entry per
decoded frame, not the future 10 Hz analysis table.

Encoded dimensions and display dimensions are reported separately. Rotation
metadata is recorded but no image rotation is performed at this loading stage.
When the last frame's duration is missing, duration uses the median frame
interval (or FPS for a single-frame video) and `duration_estimated` is true.

## Tests

```bash
uv run pytest -q
```

Unit tests cover timing, rotations, invalid inputs, sensor timestamps, and CLI
error behavior. Integration tests generate small temporary videos with FFmpeg,
decode them with ffprobe, and run the documented CLI in a separate process.
They include constant-rate and variable-rate videos plus corrupt input.
Integration tests are explicitly skipped if FFmpeg or ffprobe is unavailable.
Tests do not open `data/`, read secrets, download model weights, or call Claude.

Generated test videos are not a substitute for the two real training-clip
checks required by `AGENTS.md`. Run those before marking this stage complete.

Five public development clips are now available locally under
`outputs/public_samples/`: four EgoDex folding/cloth recordings with pose files,
and one EPIC-KITCHENS excerpt with actual gyro/accelerometer measurements.
See [sample details and commands](docs/PUBLIC_SAMPLES.md) for sources, processing,
and the distinction between poses and raw IMU. Downloaded assets are ignored by
Git; they are not automatically fetched by the tests.
