# TrueTake

Review first-person task videos for robot-training collections: keep useful footage, explain failures, and flag redundant behavior.

## Screenshot

[Placeholder: insert a screenshot of the report and its five Start-here tiles.]

## What it does

- **Quality:** checks the requested goal, visible process, and format/privacy requirements against a lab YAML contract; conservatively trims confirmed unusable intervals.
- **Verification:** compares video camera motion with optional gyroscope data.
- **Value:** ranks approved clips for behavioral variety, flags duplicates, and suggests what to film next.

Four outcomes: **APPROVED, NEEDS REVIEW, REJECTED, DUPLICATE**. A cut badge is an approved derivative; pending trimmed copies remain outside the accepted collection.

## How it checks the AI

Claude returns evidence before yes/no/unsure goal answers. A validator checks supplied frame IDs, timestamps and evidence references. Formulas decide measured rules using the lab YAML targets and review bands; Claude does not override those formulas. Unsure, unavailable or invalid goal evidence stays NEEDS REVIEW unless a separate definite failure requires rejection. Failed validation is not silently accepted. Automatic model escalation is not implemented.

## Results

Current saved run: **5 approved, 8 need review, 12 rejected, 1 controlled duplicate**; **3.3 approved usable minutes**, including one sensor-verified approved clip.

- Seven TRAIN pairings, one fixed axis mapping: real gyro scores **0.43–0.67**, swapped **0.02–0.14**. These are calibration/development results, not held-out accuracy.
- Splice rejected at **0.153 < 0.29**. Window evidence marks the cut at 0:30; windows do not determine the verdict.
- Agreed with labels on **0 of 0 comparable test clips**; NEEDS REVIEW counted separately (**2**). All 11 TEST completion labels are unknown: 0 verified, 11 unverified. An accuracy percentage cannot be claimed.

Full local audits: `outputs/final/agreement.md` and `outputs/hand_presence/REPORT.md`. The cut example is pending, and the wrong-task examples lack cached goal reviews; neither is presented as a successful approval/rejection.

## How to run

Python 3.11, WSL/Linux, FFmpeg and ffprobe; CPU only.

```bash
uv sync
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --env-file .env python batch.py /path/to/clips --contract contracts/mercor_folding.yaml --gyro-profile docs/gopro_epic_calibrated.json --detector-mode onepass --review-mode focused --export --output-dir outputs/demo
python3 -m http.server 8000 --bind 127.0.0.1 --directory outputs
```

Open **http://localhost:8000/demo/**. Put matching normalized sensors beside each video as `<clip_id>_sensors.csv`. The supplied gyro profile is for this EPIC/GoPro setup, not arbitrary phones. The standard runtime key is ANTHROPIC_API_KEY; credentials are not included. Downloaded clips/model weights are not in the code archive.

Rebuild an existing saved run without model calls: `uv run python tools/build_demo_report.py`. Tests: `uv run pytest -q`.

## Limits

Small, reused development clip set with incomplete human labels. EPIC stabilization weakens tilt correspondence. Gyro agreement is supporting evidence, not proof of authenticity. Budget roughly **2.5 min/clip on laptop CPU** for uncached processing (estimate, not a new benchmark); cached exports/rendering vary. Hand detection can miss visible hands. Task-time coverage currently uses the clip minus exclusions as a proxy. No robotic downstream training benefit has been demonstrated.

## Data credits

- **EPIC-KITCHENS-100:** [dataset](https://epic-kitchens.github.io/2024), [CC BY-NC 4.0 terms](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/license.txt). Excerpts and demo manipulations are derived material. Cite D. Damen et al., *Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100*, IJCV 130, 33–55 (2022), [DOI](https://doi.org/10.1007/s11263-021-01531-2).
- **Existing local collection / own recordings:** no author-recorded footage has been supplied. `source=own` is a catalog label, not an authorship claim. Its folding footage comes from [EgoDex](https://github.com/apple-aiml-research/ml-egodex), whose dataset terms are CC-by-NC-ND. Cite R. Hoque et al., *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video* (2025), [paper](https://arxiv.org/abs/2505.11709). Source licenses continue to apply to demo assets; they are not relicensed as project code.
- **Hub:** reserved for internal evaluation only; no clips were available in this saved run. No media or thumbnails from this source are redistributed.

## Built before vs during the event

[Fill in before submission: list what existed before the event and exactly what was built during the event.]
