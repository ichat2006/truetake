# TrueTake demo

Serve generated outputs, then open http://localhost:8000/demo/index.html:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory outputs
```

The manifest-driven report includes every available input plus the demo variants.
Hub inputs are excluded from `outputs/demo/`; the all-clips view is written to
`outputs/demo_internal/`. Missing files are listed in the saved analysis. A plain
result table is available at `demo/summary.csv`. EgoDex originals use a codec
that browsers often cannot play; the report creates small H.264 presentation
copies. Originals stay unchanged. Click thumbnails or timestamp flags to seek.
Detected-face intervals receive full-frame blur in video previews, displayed
thumbnails, and new Rerun recordings. Sampled detection is not a guarantee of
complete anonymization. Original evidence stays local and unredacted.

## Regenerate safely

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --env-file .env python tools/run_manifest_demo.py
uv run python tools/build_demo_report.py
uv run pytest -q
```

The runner reads `data/manifest.csv` and the additional rows in
`data/clips/variants/manifest.csv`; it never rewrites either input manifest.
Each completed clip is saved atomically to `outputs/manifest_run/<clip_id>.json`.
Existing result files are skipped before decoding, inference or API calls. Valid
detector/Claude caches and exports are reused; existing Rerun files are skipped.
An interrupted Rerun write stays at a temporary filename until it finishes.
The CLI lock prevents two manifest runs from processing clips concurrently.
Failed result files also remain on disk and include their error; inspect them
before explicitly removing that output result file to retry. Input data stay untouched.
No combined report is rebuilt during the run. `tools/build_demo_report.py` is the
separate aggregation/publication step, reading the per-clip results and writing
`outputs/manifest_run/batch.json`, `outputs/demo/`, and `outputs/demo_internal/`.

Run in your terminal so it survives this chat session:

```bash
mkdir -p outputs/manifest_run
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 \
  uv run --env-file .env python -u tools/run_manifest_demo.py \
  > outputs/manifest_batch.log 2>&1 < /dev/null &
echo $! > outputs/manifest_batch.pid
```

After it finishes, rebuild with `uv run python tools/build_demo_report.py`.
The calibrated profile is `outputs/profiles/gopro_epic_calibrated.json` and uses
the operator-approved 0.29 threshold from seven fixed-mapping train examples.
Folding clips use `contracts/mercor_folding.yaml`; EPIC and sensor-attack fixtures
use `contracts/wash_dishes.yaml`. The known controlled duplicate reuses its source
review and is not a second training example.

The five fixtures are created once by `tools/make_demo_variants.py`. That tool
refuses to overwrite existing files. Their verified labels describe the injected
transformation, while expected outcomes are kept separate from observed verdicts.
An unmet expectation is not silently converted into a matching decision.

The presentation uses a large summary, a thumbnail map, verdict badges, original
and output players, and actual applied cut segments. Scores, source metadata,
model evidence and manifests stay in collapsed details. Publication copies its
playback assets into the report directory so `data/clips/` inputs do not produce
broken URLs under the `outputs/` HTTP server. Task approval and motion verification
remain separate; missing motion data does not automatically reject a clip.

The runtime uses standard `ANTHROPIC_API_KEY`. With the operator's permission,
uv loads the local environment file for this command; application code does not
open it. There is no need to rename the key. Never paste it into a contract or
prompt. `.env`, `.env.*`, `outputs/`, and model assets are ignored by Git.
No command here stages, commits, or pushes anything.

Without credentials, omit `--env-file .env`: matching validated cached reviews
still work; uncached reviews remain pending. Caches contain public-clip evidence,
model responses, model name, prompt version, and token counts, never API keys.
The default API model is `claude-haiku-4-5`; `--claude-model` selects another
available model. Tests make no live API calls.

Processing is sequential and CPU only. MediaPipe samples at 5 Hz, CLIP at 2 Hz
with batches of eight and two torch threads. Detector caches use source content,
detector code, model assets, and the dependency lockfile. Contract/prompt changes
reuse detector results but invalidate AI reviews. The initial AI review receives
eight small selected images in the default focused mode. Selection preserves
the first and last frame, nearby start/end context, and coverage of each
temporal quarter. Darkness, low image detail, and little image change produce
inspection hints, not task verdicts or permission to cut. Hands are not a gate.
Claude determines task relevance and completion for each new unique recording.
At most one follow-up call adds up to two eight-second windows at 2 Hz when
required evidence is unresolved. Optional hand/idle questions do not trigger
another call after required judgments are decided. Deferred questions remain
in the response JSON. Invalid responses remain inconclusive.

This implements the Claude evidence-selection part of Plan A. Local detectors
use separate passes by default. The opt-in one-pass detector below shares one
analysis decode and adds conservative local gates. No gyro calibration or
authentication claim is added.

Use `--review-mode baseline` to compare against the original 12-image review;
`--review-mode focused` is the default. Prompt/model/contract/evidence changes
invalidate the AI cache. Exact matching reviews can be reused without a call.
Saved JSON cards contain selected-image counts and actual new input/output tokens;
Rerun logs selected-frame timestamps under `review/claude_evidence` and local
candidate windows under `review/claude_candidates`. Follow-ups can increase
total images and tokens, so compare total usage, not just the initial budget.
Saved model provenance includes tokens from the original review; per-run usage
counts only newly completed calls. Validation failures still consume tokens.

To regenerate the current comparison without new calls when caches match:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python batch.py outputs/public_samples --contract contracts/mercor_folding.yaml --review-mode focused --compare-to outputs/comparisons/baseline_batch.json --output-dir outputs/demo --include-duplicate
```

The saved five-clip development comparison matched all five baseline verdicts.
Focused reviews used 40 initial images versus 60; one required follow-up brought
total images to 76 versus 60 and calls to six versus five. Total input tokens
fell from 20,646 to 20,157 (2.4%); output tokens fell from 10,313 to 9,659 (6.3%).
These are modest measured savings, not evidence of large-scale savings or
held-out accuracy. The controlled duplicate is excluded from the comparison.

## One-pass local detectors

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python batch.py outputs/public_samples --contract contracts/mercor_folding.yaml --detector-mode onepass --review-mode focused --compare-to outputs/comparisons/multipass_focused_batch.json --output-dir outputs/demo --include-duplicate
```

The comparison snapshot is the previous focused review with multi-pass detectors.
For new uncached clips, add `--env-file .env` after `uv run` to enable Claude.
Use `--detector-mode multipass` for the original implementation. Baseline
12-image reviews currently require that multi-pass detector mode.

One-pass detectors compute basic image measurements at 10 Hz, faces and hands
at 5 Hz, fingerprints at 2 Hz, and CPU-batched CLIP embeddings at 1 Hz instead
of 2 Hz. Hands are not a CLIP gate: other tasks and brief hand exits retain
scene context. Only nearly black images (mean intensity below 5/255 AND maximum
intensity at most 10/255) skip hands/CLIP. Skipped measurements remain unknown;
they are never interpreted as zero hands or proof of task failure. Faces still
run at every scheduled 5 Hz timestamp, including nearly black frames.

The loop temporarily spools 320-pixel JPEGs under the clip output directory,
then retains the selected eight and removes the spool. This bounds RAM and
avoids a second source-video decode just to select evidence. Small overview
thumbnails also remain for inspection. Originals are unchanged. Detector
inference batches contain at most eight CLIP frames. One-pass caches have
separate filenames and fingerprints from the multi-pass caches.

The batch JSON's detector totals contain original inference counts for independent
recordings; cached reruns and controlled reuploads do not add inference work.
Per-clip `detector_stats` includes stage timing and skipped-frame counts. One
analysis decode does not include ffprobe's timestamp read, file hashing, a
requested Claude follow-up, or the separate 960-pixel Rerun rendering pass.
No unvalidated task classifier, frozen-frame rejection, face rejection, or
early clip termination is introduced.

The five-clip development check preserved all five verdicts and reused the same
Claude evidence with zero new API calls. CLIP inference decreased from 319 to
161 frames (49.5%). Face and hand inference each covered 793 frames in both
implementations; all clips were above the nearly-black gate. Basic measurements,
hand counts, face confidences, fingerprints, and CLIP embeddings at shared
timestamps matched the preserved multi-pass results. Dark-gate behavior is
covered by synthetic tests. A guarded cache replay verified that neither model
initialization nor API calls occur when the corresponding caches match.

The measured one-pass detector time was 124.46 seconds across the five clips,
excluding CLIP weight loading and later pipeline stages. A comparable baseline
wall-time benchmark was not recorded, so do not claim a measured speedup ratio.
The detailed snapshot is `outputs/comparisons/onepass_vs_multipass.json`.

The demo HTML omits the collection-summary, detector-funnel and Claude-comparison
blocks. Diagnostics remain in the JSON artifacts. The page focuses on the clip
map, review decisions, evidence and corrected outputs. The map uses approximate
pairwise appearance/action similarity; its axes have no physical units or quality
meaning. Status determines color. Exact duplicates can overlap.

## Supply the lab's collection request

Edit a copy of [the active YAML contract](../contracts/mercor_folding.yaml), or
pass explicit collection instructions:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --env-file .env python batch.py outputs/public_samples --contract contracts/mercor_folding.yaml --instructions "Partial folding demonstrations are allowed; useful folding motion must still be visible." --output-dir outputs/partial_demo
```

The active policy reflects the operator's prototype decisions, not official
Mercor rules. Completion and understandable folding action are required by
default. Both hands are optional; a brief hand exit is not a rejection. The
task-visibility gap allowance is two seconds. Faces trigger blur rather than
whole-clip rejection. There is no arbitrary minimum clip duration.

Completion can be relaxed by an explicit collection request; unrelated cooking
or merely holding an already folded item still fails the task-action requirement.
Required goal failure rejects; uncertain required evidence needs human review.
Optional warnings cannot override a successful required task assessment.
The evidence-first prompt is [review_v2.md](../prompts/review_v2.md).

Only densely supported follow-up intervals become exclusion suggestions.
Cuts shrink inward by one second at each boundary and preserve completion
evidence. Uncertain spans remain intact. `excluded_intervals` are the compiled
suggestions; the optional corrected-video export below applies supported cuts.
Usable seconds are estimates from sampled evidence, not continuous annotations.

## Export the corrected collection

Export from saved reviews without repeating detectors or Claude calls:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python export.py outputs/demo/batch.json --output-dir outputs/demo/exports --rerun
```

`--rerun` refreshes the source Rerun recordings with edit-map annotations.
Omit it to export videos and update the HTML report without rendering Rerun
again. To export during a new pipeline run, add `--export` to `batch.py`.
No API key is needed when exporting saved decisions.

The export folder contains `collection.json` and versioned per-clip folders:

- `accepted/`: copies whose source review passed and whose privacy edits have
  usable face boxes. Rejected recordings and controlled demo reuploads are omitted.
- `pending/`: unapproved copies retain the full source timeline. No requested
  cuts are applied. A privacy request without usable face boxes produces a
  full-frame-blurred pending copy rather than a training-eligible export.

Each exported clip includes H.264 video, an edit manifest, and a frame CSV
mapping every output frame/time to its source frame/time. When motion data is
available, a paired `_sensors.csv` contains `t,gx,gy,gz,ax,ay,az,source_t,segment_id`.
`t` uses the edited video's clock; `source_t` and `segment_id` retain provenance.
Values stay in rad/s and m/s² in the original device axes. Samples outside kept
intervals are removed, with no interpolation or filling across edit joins.
Do not integrate across joins without resetting; the manifest lists join times.
Existing recording clock offsets are preserved, not estimated or calibrated.

Only accepted, compiled exclusions can become cuts. Boundaries snap inward to
source frame times, completion evidence/context is protected, and an empty
video is refused. VFR presentation times and the final frame's hold duration
are preserved; exported frame counts, timestamps and duration are checked to
0.1 ms before publishing the files. Display rotation is baked into pixels once;
gyro axes are not remapped. Source resolution is kept, with at most one padding
pixel per dimension for H.264. Derivatives are 8-bit, CRF 18, and silent; original
audio remains in the untouched source. Originals are never replaced.

Face boxes are conservatively combined/padded within detected privacy intervals,
so blur can cover more than the face but preserves pixels elsewhere. Detection
is sampled and is not a complete anonymization guarantee. Post-edit task quality
is not automatically reassessed. Edited files are explicitly marked derivatives;
source authentication/verification does not transfer to them.

The report adds corrected players, download links and source/edited seek buttons.
Rerun keeps its original seconds timeline and logs kept/cut mappings under
`review/edits`. Reopen a recording after regeneration to see updated annotations.
Fingerprints and artifact hashes reuse matching exports. Changed plans write a
new version folder; old versions and source recordings remain intact. The
collection manifest identifies the current files.

Exports are encoded with `libx264`, `yuv420p` and `+faststart`; the exporter probes
the actual output and records its codec in each manifest. To regenerate videos
and refresh the report even when an export cache matches:

```bash
uv run python export.py outputs/demo/batch.json --output-dir outputs/demo/exports --force
```

Reload `http://localhost:8000/demo/index.html` after regeneration to use the current
versioned video links. Add `--rerun` only when the Rerun annotations also need refreshing.

## Present the results

Show a folding clip and its required-rule reasons, then the cooking clip under
the same contract. A contract score is the percentage of decided required rules
that pass, not acceptance confidence. Unknown evidence stays a review case.
Scene distinctness is a separate relative batch score, not training value.

The map includes all clips for inspection. Training-value ranking and redundancy
use accepted clips only. Available scene and Claude strategy views are combined;
hand motion is omitted when continuous tracking coverage is insufficient.
The controlled reupload is clearly labeled: it reuses one recording's evidence
and is not independent footage. It demonstrates duplicate detection without
another download or API call. Omit `--include-duplicate` for the five originals.

Every card links its JSON, frame table, and Rerun recording. To inspect another
recording, start the browser server with that file (stop the old server first
if reusing the ports):

```bash
RAYON_NUM_THREADS=2 uv run rerun --serve-web --bind 127.0.0.1 --web-viewer-port 9090 --port 9876 --server-memory-limit 256MB outputs/demo/basic_fold_1/timeline.rrd
uv run rerun rrd verify outputs/demo/basic_fold_1/timeline.rrd outputs/demo/P01_102_000_060/timeline.rrd outputs/demo/batch_map.rrd
```

Open http://localhost:9090. No native viewer or Gyroflow is launched.
Sensor verification is deliberately inconclusive while calibration is deferred;
missing sensors say `no motion data`. Neither rejects under the active contract.
Hardware authentication has not been established.

Human labels are still needed before reporting accuracy:

```bash
uv run python evaluate.py outputs/demo/batch.json outputs/human_labels.csv --split train
```

Labels need `clip_id,split,goal_met`. Never tune on test clips. Sample provenance
and licenses are in [PUBLIC_SAMPLES.md](PUBLIC_SAMPLES.md).
