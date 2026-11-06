# Reproducible development corpus

Run from the repository root. No new project dependency is required. HF downloads
use an isolated `uvx --from huggingface_hub==2.4.0 hf` CLI. Videos use two FFmpeg
threads, CPU H.264, original resolution, CRF 23, and no audio.

## Current resolved set

`data/manifest.csv` describes **planned and existing** files, not a claim that
every file has been downloaded. `tools/clip_sources.json` pins exact remote
filenames, revisions, intervals, and annotation IDs.

- EPIC: 16 one-minute windows; eight participants, two per participant; 12
  washing and four cooking windows with no overlapping wash narration. Participants
  P01, P03, P04, P06, P07, P22, P28, P30. P02 did not supply two windows meeting
  the fixed minimum of ten wash-annotated seconds, so P07 replaces it.
- Hub: four Fold Clothes recordings plus one each of tea, wiping and tidying.
  These are complete source videos, including two ten-minute folding clips.
- Existing collection: six entries referencing five actual MP4s under `outputs/public_samples`, labelled
  `source=own` per the requested local-collection convention. Their notes retain
  actual EPIC/EgoDex provenance and licenses; this does **not** claim authorship.
  The sixth entry is a controlled reupload alias, referencing `basic_fold_0` in
  train. It is not an independent source recording or an additional training example.
- HA: **blocked**. The supplied repository returns 401/not found even through the
  CLI's existing authentication. Its publicly indexed card describes gated
  access requiring acceptance of contact-sharing terms. The documented layout
  and feature reference are available, but no actual parquet has been inspected.

The requested split has **18 train / 11 test** entries (62/38, counting the
controlled alias). EPIC stays split by participant (five train / three test people).
Hub `#41`, `#42` (folding), and `#09` (wiping) are test; `#01`, `#02`, `#03`, `#13`
are train. Own `basic_fold_1` and `fold_stack_unstack_unfold_cloths_1` are test;
all other own entries are train. Whole files never cross splits. Hub/EgoDex actor
identities are unknown: this clip-level split does not establish participant
independence. The own test clips were used previously for development, so these
assignments do not establish a pristine held-out evaluation.

`task_expected` is a source annotation, **not** the requested folding contract.
`cooking_no_wash` means no wash annotation overlaps the selected interval, not
that visual absence was established. Every `goal_met_expected` is `unknown`:
task labels do not prove completion. Every label remains `unverified` for manual
review. There are no fabricated goal failures. Only the existing controlled alias
has `attack=controlled_reupload`; it stays with its original in train.

## Commands

Preparation downloads only small annotation/license metadata and creates the
manifest and plan. Identical files are reused; differing existing outputs cause
an error. Never regenerate into an existing manifest to change labels.
The user-authorized correction used `apply-requested-split`; it backs up the
previous manifest/plan under `outputs/corpus_backups/`, changes only the manifest
and plan, and refuses to discard reviewed labels. Default `prepare` reproduces
the corrected split for a fresh checkout.

```bash
uv run python tools/fetch_clips.py prepare
uv run python tools/fetch_clips.py status
```

One EPIC smoke test (the only video download run during implementation):

```bash
uv run python tools/fetch_clips.py fetch --source epic --clip-id P07_111_000_060 --limit 1
```

Download all **currently resolved EPIC and Hub** entries:

```bash
uv run python tools/fetch_clips.py fetch --source all --accept-hub-evaluation-license
```

This deliberately returns exit status 2 after downloading the resolved entries,
because HA is still blocked. It is **not yet an
all-three-datasets downloader**. To download EPIC alone, without Hub's license gate:

```bash
uv run python tools/fetch_clips.py fetch --source epic
```

Existing outputs are accepted only when their provenance and SHA-256 hashes match
the pinned job. Partial or unrelated existing files cause an error; nothing is
overwritten. New full-length EPIC videos live in fresh per-run raw directories,
are shared across selected cuts from that source, and are deleted after successful
processing. Source gyro/accel files and HF metadata are retained. Downloads
interrupted before finishing may leave files; they are never deleted by later runs.
FFmpeg outputs are hardlinked into clips/ without a second video copy.

The first full download can transfer many gigabytes even though EPIC outputs are
only 16 minutes: these are full source recordings. Processing one source at a
time limits disk usage. HF authentication is optional for the public EPIC mirror;
the CLI can request login for rate limits. Never paste credentials into commands,
manifests, the repository, or chat.

## Sensor normalization and motion screening

EPIC CSV columns are `Milliseconds,GyroX,GyroY,GyroZ` and
`Milliseconds,AcclX,AcclY,AcclZ`. Divide milliseconds by 1000 and subtract the
selected window start, preserving any initial sub-frame/sample delay. Published
GoPro SI gyro and acceleration values remain rad/s and m/s²; physical X/Y/Z are
preserved, with no camera-axis permutation or sign guess. Interpolate acceleration
to gyro times strictly within shared coverage; refuse gaps over 100 ms. The end
is exclusive. Timestamps remain strictly increasing.

The adapter warns when 99th-percentile component magnitudes exceed 20 rad/s,
which may indicate degrees/s or extreme motion. It does not guess a conversion.
Camera stabilization, video rotation and visual/IMU clock calibration are **not**
solved by converting units. These are real but uncalibrated IMU measurements.

The fixed acquisition screen requires gyro axis-standard-deviation norm at least
0.015 rad/s, and median one-second image change at least 1 grayscale level at
160x90. It is an acquisition heuristic, not proof of visible head rotation or a
trained quality threshold. A failing window is reported rather than silently
substituted under the same manifest ID. Review motion visually before using
these clips to calibrate verification.

## HA access and mandatory schema checkpoint

The indexed v1 card documents one shared `data/chunk-000/file-000.parquet`, not
one sensor parquet per episode. Head video paths use
`videos/observation.images.egocentric/chunk-000/file-{episode:03d}.mp4`.
Folding episodes are 11, 15–19; mixed folding/cleaning episodes are 3 and 8.
Adding two cleaning negatives therefore means ten episodes, not eight.
These are published descriptions, not verification of the inaccessible files.

```bash
uvx --from huggingface_hub hf auth login
uv run python tools/fetch_clips.py inspect-ha
```

The second command lists the repository tree, then **stops**. Once accessible,
first select the actual head video and sensor parquet paths for the requested
episodes, pin the revision, and download only those files. Then run:

```bash
uv run python tools/ha_to_csv.py path/to/episode_sensor.parquet
```

This prints the Arrow columns, types and metadata without changing the parquet.
**The HA conversion adapter is not implemented**: it must wait for this checkpoint.
Need confirmation of head sensor identity, gyro/accel columns or nested fields,
timestamp units, video/sensor time origin, rad/s versus deg/s, m/s² versus g,
gravity inclusion, and axis definitions. Pose columns must not be substituted for
raw IMU. The task requires stopping before guessing these facts.

## Licenses and sharing

- EPIC: CC-BY-NC-4.0. Preserve attribution; the manifest marks re-encoded excerpts
  as derivatives. Official annotation revision is pinned separately from the mirror.
- Hub: **evaluation of potential data use only**. No redistribution or rehosting
  of source data **or derivatives**, and no reidentification. All Hub manifest
  rows explicitly say evaluation-only / not training-eligible. Downloading them
  does not grant permission to train a model, export them in a shareable collection,
  or publish them in a demo. Get terms from Hub for additional uses.
- Existing EgoDex recordings retain their prior CC-BY-NC-ND provenance.

The current batch/exporter does **not** enforce this new manifest's license tags
or train/test split. Do not point it at the whole corpus and publish the exported
collection. This task adds an acquisition manifest, not a licensing-policy change.

Sources:
[EPIC mirror](https://huggingface.co/datasets/awsaf49/epic_kitchens_100),
[official annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations),
[GoPro units](https://github.com/gopro/gpmf-parser),
[Hub license](https://huggingface.co/datasets/Hubdata/egocentric-iphone-rgb/blob/a90d2d89eb1cd4ca824d23d4dfdddc4edf463e08/LICENSE).

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q
uv run python cli.py data/clips/epic/P07_111_000_060.mp4 --sensors data/clips/epic/P07_111_000_060_sensors.csv --contract contracts/fold.yaml
uv run python cli.py outputs/public_samples/egodex/basic_fold_1.mp4 --contract contracts/fold.yaml
```

These are loader smoke checks, not Claude quality judgments. The first must show
motion data loaded, and the second must show `no motion data`. No API calls are
needed. Repeating the exact one-clip fetch must print `CACHED` with no download.
