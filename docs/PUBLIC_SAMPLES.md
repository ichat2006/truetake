# Small public development sample

The clips are kept in `outputs/public_samples/`, which is ignored by Git.
Nothing was downloaded into or changed under `data/`. These are development
fixtures from the source datasets' training splits, not a held-out evaluation set.

## Selected recordings

| Local path below `outputs/public_samples/` | Source | Duration | Associated motion data |
| --- | --- | ---: | --- |
| `egodex/basic_fold_0.mp4` | EgoDex training part 2, `basic_fold/0` | 9.27 s | Matching `.hdf5` poses |
| `egodex/basic_fold_1.mp4` | EgoDex training part 2, `basic_fold/1` | 20.07 s | Matching `.hdf5` poses |
| `egodex/fold_stack_unstack_unfold_cloths_0.mp4` | EgoDex training part 2, `fold_stack_unstack_unfold_cloths/0` | 28.67 s | Matching `.hdf5` poses |
| `egodex/fold_stack_unstack_unfold_cloths_1.mp4` | EgoDex training part 2, `fold_stack_unstack_unfold_cloths/1` | 40.17 s | Matching `.hdf5` poses |
| `epic_kitchens/P01_102_000_060.mp4` | EPIC-KITCHENS-100, P01_102, first 60 seconds | 60.00 s | Actual gyro and accelerometer samples |

Total: five clips, approximately 158 seconds (2 minutes 38 seconds).
The shortest recording is below the example contract's 10-second minimum;
keep it as a useful short-input fixture, not a presumed positive example.
Task folder names are source annotations, not manually verified quality labels.

## EgoDex

[Official repository and dataset documentation](https://github.com/apple-aiml-research/ml-egodex).
[Training archive, part 2](https://ml-site.cdn-apple.com/datasets/egodex/part2.zip).

Only the four selected MP4/HDF5 pairs were extracted using HTTP byte ranges;
the full archive was not downloaded. Videos are original, unmodified members.
`egodex/manifest.json` records exact archive paths, file sizes, SHA-256 hashes,
and source split.

The HDF5 files contain ARKit-estimated camera/body/hand poses aligned to video
frames. They are **not raw gyro or accelerometer measurements**. Do not pass
them to `--sensors` or fabricate IMU readings from them for verification.
The current loader processes these clips with `imu=None`; an HDF5 pose adapter
is not implemented. No HDF5 dependency was added.

The official repository states that the dataset is under CC-BY-NC-ND terms.
Retain source attribution and consult those terms before redistribution or
commercial use. The dataset license is distinct from the code license.

## EPIC-KITCHENS

[Official dataset page](https://epic-kitchens.github.io/2024).
[Official download script and paths](https://github.com/epic-kitchens/epic-kitchens-download-scripts).
[Split list](https://github.com/epic-kitchens/epic-kitchens-download-scripts/blob/master/data/epic_100_splits.csv).
[Dataset license](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/license.txt).

P01_102 is in the action-recognition training split. Its first minute is a
kitchen-activity fixture, not a positive T-shirt-folding example.

The excerpt is H.264 re-encoded at CRF 26, preserving 1920x1080 resolution and
50 fps, with audio omitted. It is deliberately marked as a derived excerpt;
do not use it as ground truth for an "unedited original" check.

The accompanying files are:

- `P01_102_000_060-gyro.csv`: source gyro rows covering 0–60 seconds.
- `P01_102_000_060-accl.csv`: source accelerometer rows covering 0–60 seconds.
- `P01_102_000_060_sensors.csv`: normalized `t,gx,gy,gz,ax,ay,az` input.
- `manifest.json`: original URLs, source interval, hashes, processing, and caveats.

Normalization divides `Milliseconds` by 1000, preserves the published gyro
axis columns, and interpolates acceleration to gyro timestamps strictly inside
their shared coverage. No time shift, axis permutation, or extrapolation is
applied. Units use the [GoPro GPMF SI convention](https://github.com/gopro/gpmf-parser#imu-orientation-uses-these-axis-labels):
gyro in rad/s and acceleration in m/s². Axis mapping, capture stabilization,
and empirical video/gyro clock alignment still need checking before using
these samples to calibrate verification thresholds.

## Run the loader

```bash
uv run pytest -q
uv run python cli.py outputs/public_samples/egodex/basic_fold_1.mp4 --contract contracts/fold.yaml
uv run python cli.py outputs/public_samples/epic_kitchens/P01_102_000_060.mp4 --sensors outputs/public_samples/epic_kitchens/P01_102_000_060_sensors.csv
```

The first clip should report `sensor_status: "no motion data"`; the second
should report `sensor_status: "loaded"`. Both correctly report
`review_status: "not_run"`: this stage tests ingestion, not task completion or
authenticity. Machine-readable smoke-test results are saved beside the samples.
