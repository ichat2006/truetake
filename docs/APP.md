# Local TrueTake app

Run from the repository root (Python 3.11, WSL, FFmpeg/ffprobe):

```bash
uv sync
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 uv run --env-file .env uvicorn app.main:app --port 8001
```

Open http://localhost:8001 . Stop your terminal server with Ctrl+C. The key is consumed by the runtime; application code never opens or prints the environment file. The app binds to the local machine; it is not a hosted multi-user service.

## 60-second demo path

1. **0–15s / Define the task:** load the folding example; point to the goal, process targets, intended use and live YAML. Keep the example unchanged for saved AI evidence reuse.
2. **15–30s / Add footage:** Load demo set. Point to video-only versus motion-data tags and licence/credit lines. Dataset licences are locked; no restricted clips or EgoDex variants enter the demo set.
3. **30–45s / Reviewing:** Review clips. Stage chips reflect completed/cached work, and counters update from actual results. The local demo uses saved detector/AI evidence and makes no model calls. The first run may take longer to copy or encode exports; warm it once before screen recording.
4. **45–60s / Results:** show the eligibility line separately from quality, then inspect the Start-here tiles and full report. An unknown goal is not turned into a pass. No approved cut example is claimed when none exists.

## Uploading your own footage

Drop multiple videos and optional matching `<video_stem>_sensors.csv` files. Raw uploads remain under `outputs/uploads/<run_id>/raw/`; analysis and reports stay under that run. The default licence is unknown (analysis only, no preview/export). Select the licence you actually have; byte-identical known dataset content retains its catalogued licence.

CPU jobs run sequentially in a background thread. Polling reports progress; a processing exception becomes NEEDS REVIEW with its error in details. Content hashes reuse prior uploads and existing catalogue caches. Changed sensor files must be reverified, even if the video matches. The EPIC camera/gyro profile is not blindly applied to new phone footage; without a matching calibration, it remains unverified.

Demo mode never pays for a fresh goal review. Changing the goal makes incompatible saved goal evidence unavailable. Actual uploads can use the existing Claude model and prompt when a runtime key is supplied. Numeric measured rules still use the existing contract formulas. Item-visible percentage coverage is not calibrated by the current detectors: requesting it as a numeric rule stays unverified. Legacy hard checks for slow motion, editing and tripod state may remain inconclusive; they are not silently passed.

Licence policy lives in `contracts/licenses.yaml`; true origins are in local `data/manifest_v2.csv`. Original-only media are not trimmed, blurred or re-encoded. Proposed cuts are shaded on originals; eligible licences can produce a proposed-cut derivative. Public outputs and the demo set exclude sources whose policy disallows sharing. Read `docs/LICENSING.md` for the full audit.

## Validation

`uv run pytest -q` runs API, upload-pairing, licence isolation, cache, error-result and pipeline tests. The real Chrome walkthrough is `tools/app_browser_smoke.cjs`, using the installed Windows Node and Chrome without npm packages. It records 1920×1080 screenshots of all four screens and a result JSON under `outputs/app_screens/`. No keys are written into screenshots or responses.

The local cached corpus/model weights are intentionally not committed. A fresh checkout can upload files, but needs the existing pipeline model assets to process uncached footage; Load demo set requires the local permitted catalogue.

Run state is in memory; outputs persist on disk. After a server restart, start a new review rather than expecting an old browser session to resume. Cached compatible work is reused. Original dataset licences are locked even when the same bytes are uploaded under a different filename.

## Licensed cut fixture

`uv run python -m tools.make_screenrecord_cut` creates a permitted EPIC black-gap fixture under outputs. Its extra manifest row is in `outputs/manifest_v2_additions.csv`, merged with the immutable `data/manifest_v2.csv`. The parent washing goal evidence is reused only for this known insertion, with evidence timestamps and frame IDs shifted and validated. Sensors and sensor-verification claims are removed. Basic encoded-frame image statistics are measured locally; no new model calls are made. The mixed-task demo labels this fixture with its saved washing checklist. Existing conservative trimming retains one second at each gap edge: the full 8–13 s defect is shaded, while the darker inner band shows the actual cut. Any NEEDS REVIEW verdict is retained.

## Examine

Every result card opens the installed Rerun web viewer inside a full-screen modal. Its assets are served locally from the installed wheel and pinned to exactly the installed SDK version (currently 0.42.0). Missing recordings are generated on demand from saved tables; this makes no model calls. Sensor timelines include mapped pan/roll against gyro with the stored clock offset applied. First opening a long timeline can take several seconds to load.

EgoDex originals in this catalogue use the MPEG-4 codec, unsupported by Rerun video playback. Their original-only licence forbids video conversion: Examine shows a labelled static midpoint preview plus the cached signal timeline. The EPIC splice and black-gap example have playable timeline images. No restricted media is exposed through the Examine endpoint.

## Optional missing-goal review (you run this)

This final pass made no API calls. Four catalogue evaluations still lack saved goals: `P01_102_000_060_wrong_task`, `P06_102_575_635_wrong_task`, `basic_fold_1_black_gap`, and `basic_fold_1_freeze`. Only the first is in the curated demo. Keep them unconfirmed, or explicitly run this to review only missing goals with the existing model/prompt and saved detector tables, then rebuild the reports:

```bash
uv run --env-file .env python -m tools.review_missing_goals --execute
```

That command can incur API cost; it was not executed during the cache-only final pass. Restart the app review afterwards to load the updated results.

## Time-saved estimate

The results panel defaults to **2.0 minutes per clip** and **$25/hour**, both explicitly labelled as assumptions and editable. Human hours without TrueTake are all clips × minutes/60; with TrueTake they are unresolved clips × minutes/60. Approved/rejected clips are counted as automatic decisions; duplicates are shown separately and require no repeat full review. Unattended computation time is excluded. Dollar savings subtract logged AI cost when available. A cached run has zero new calls/tokens and $0 new AI cost; unpriced live calls display an unknown cost rather than a fabricated price. The per-1,000 projection assumes the same review and cache-use rates; it is not an accuracy or throughput benchmark.
