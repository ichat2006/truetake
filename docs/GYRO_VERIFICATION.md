# Gyro verdicts and train-only evaluation

`measure_alignment` retains its correlation calculation, smoothing, lag search,
axis permutation, signs, and aggregate score. Only its calibration-status gate
and two-axis lag-consistency decision changed. A weak third axis no longer vetoes
the two strongest scored axes.

`--diagnose-gyro` fits a provisional mapping over the entire clip. It derives the
reference offset from existing lag curves, using only axes with peak correlation
at least `MIN_AXIS_PEAK = 0.5`. The diagnostic reference can differ from the
unchanged aggregate-score optimum. Both are recorded; they are not interchangeable.
Five-second windows start every second across the whole recording. A window is
`insufficient_motion` when its largest physical gyro-axis standard deviation is
below `MIN_WINDOW_GYRO_STD = 0.02 rad/s`, or its best lag hits ±2 seconds. These
windows do not enter mapping, offset, or axis agreement counts. Overlapping
windows are development diagnostics, not independent validation.

## Profiles

Profiles remain schema version 1. A provisional profile cannot contain a
threshold. A calibrated profile must provide a signed permutation of the gyro
axes, a finite `min_match_score` in [0, 1], and a nonempty list of unique
`train_clip_ids` identifying the recordings that set the mapping/threshold.
Merely omitting `status` no longer enables a pass.

Example format (mapping and threshold below are illustrative, not fitted data):

```json
{
  "schema_version": 1,
  "status": "calibrated",
  "axis_mapping": {
    "flow_pan": {"axis": "gyro_gz", "sign": 1},
    "flow_tilt": {"axis": "gyro_gy", "sign": 1},
    "flow_roll": {"axis": "gyro_gx", "sign": 1}
  },
  "min_match_score": 0.6,
  "train_clip_ids": ["known_good_train_clip", "other_participant_train_clip"],
  "source_rotation_deg": 0
}
```

Use a mapping appropriate to the capture device/orientation. The loader validates
the shape and provenance fields; it cannot prove that the supplied IDs or labels
are truthful. The evaluation tool additionally rejects profile training IDs that
are not sensor-equipped train entries in its manifest.

## Train experiment

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python tools/gyro_swap_eval.py
```

Outputs: `outputs/gyro_swap_eval.csv`, `.png`, and `.json`. The JSON records
participant pairings, skipped train entries, the exact clips used, and the
threshold suggestion. Test rows are excluded before any referenced file is
resolved or opened; controlled attacks and unknown participant IDs are excluded.
Missing train downloads are reported, not silently treated as scored examples.

Every video is scored against its own sensor file and a different participant's
sensor file. Without a profile, its mapping is fitted on the real pairing and
locked for the swap; a swap cannot search for fresh signs. The suggestion is
midway between the lowest real and highest swapped score when separated. If
groups overlap, the tool prints their median midpoint but explicitly marks it
unusable as a clean separating threshold. This does not create a calibrated profile.

To evaluate one fixed mapping across all pairings:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python tools/gyro_swap_eval.py \
  --gyro-profile outputs/gyro_whole_clip/P01_102_000_060_gyro_candidate.json \
  --output outputs/gyro_swap_eval_fixed.csv
```

Per-clip mapping scores must not be presented as validation of a single shared
device profile. A few development examples are not held-out accuracy. Fixed
profiles still need sufficient motion, overlap, a non-boundary offset and
agreement between their strongest axes before the checker emits PASS/FAIL.

## Batch

```bash
uv run python batch.py path/to/clips --contract contracts/mercor_folding.yaml \
  --gyro-profile path/to/calibrated.json --output-dir outputs/verified_batch
```

Verification runs independently of task quality, including on clips ultimately
rejected by task checks. It is recomputed on cache replay using the requested
profile. A confirmed mismatch rejects the clip and clears approved usable
intervals; a task rejection is never overturned by good gyro agreement.
Missing sensors remain inconclusive with the exact reason `no motion data`.
Without a calibrated profile, sensor verification remains unverified. This task
does not solve video stabilization, rolling shutter or camera/IMU extrinsics.
