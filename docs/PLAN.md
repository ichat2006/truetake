# Mercor Hack Day: Build Plan

Nov 1, 2026 · @Ishaan Chaturvedi

## The project

TrueTake (working name) tells a robotics lab which first-person clips to accept and which are worth paying for. The demo task is folding clothes, using footage you record yourself. Every clip passes through three layers:

1. **Quality:** does it meet the lab's collection spec?
2. **Verification:** is it a real, unedited recording whose video agrees with the phone's own motion sensors?
3. **Value:** does it add a new way of doing the task, or repeat what the dataset already has?

- **Why it matters:** collectors are paid only for approved hours, and rejected footage isn't paid ([job post](https://www.workana.com/en/job/egocentric-video-data-collection)). Volume is huge: Micro1 alone reportedly collects 160,000+ hours of chore video a month ([source](https://ai2.work/blog/the-robot-data-gold-rush-filming-human-labor-to-train-machines)). More footage of the same few people in the same setting doesn't help a robot generalize; diverse footage does ([Encord](https://encord.com/blog/egocentric-data-collection-for-robotics/)).
- **Why Mercor cares:** Mercor recruits US-based contributors to film household tasks with a head-mounted phone, including folding laundry ([listing](https://hackajob.com/job/dc5bf67f-760d-11f1-a7b8-0a05e249917d-household-activity-video-contributor-us-based-train-ai-models-part-time)). It is also hiring a Software Engineer, Robotics Data, to turn field recordings into validated datasets ([posting](https://wellfound.com/jobs/4523083-software-engineer-robotics-data)).
- **Why you:** sensor fusion and clock sync (Fusion in Motion, Moog), computer vision (MALD, Johns Hopkins), and honest evaluation (AAPL forecaster).

**Pitch line:** "Labs pay for hours of footage, but most of those hours show the same motions again. TrueTake checks that each clip is real and usable, then finds the ones that teach a robot something new."

## What the user sees

Point TrueTake at a folder of clips and get back the report a lab's data team would send a collector.

- **In:** a folder of clips (each a video, plus a sensor file when the phone recorded one), and a short spec file with the lab's rules and the variations it wants.
- **Out:** a batch report, a per-clip card with timestamped reasons, a Rerun timeline for any clip, and a map of all clips grouped by folding style.

Example batch report (illustrative numbers):

```text
TrueTake batch report: folding, 52 clips, 31.4 min submitted

ACCEPTED        41 clips, 22.6 min usable (idle and off-task seconds removed)
REJECTED         7 clips: hands out of frame (3), too dark (2), vertical video (2)
VERIFICATION     2 clips failed: sensor file does not match the video
UNVERIFIED       2 clips: no sensor data (video-only camera)

VALUE
  Redundant      24 accepted clips repeat the standard fold
  Most valuable  clip_031 only fold-in-thirds; clip_044 sleeve drop and recovery
FILM NEXT        hoodie (0 clips), folding on a bed (0), left hand leading (1)
```

## Architecture

&#91;embedded content: TrueTake architecture · three layers, one report\]

Build it top to bottom: each box is one file and one Claude prompt. Every check reads the same `Recording`, defined in the brief under Building with Claude. Value scores only the clips that pass quality, and every layer's reasons land in the report.

## Pipeline, Plan A: one pass, built to scale

Read each video once at low resolution, and only run expensive models on frames that pass the cheap checks. Decoding video is cheap; the real cost is models run on frames, and Claude most of all.

1. **Before decoding (instant).** Read the file details with ffprobe and apply the hard rules. Then pre-scan the sensor file, which is only kilobytes: a phone that never moves sat on a surface or tripod (reject), plus tilt per second and hints of idle stretches.
2. **One pass at 10 frames per second, 320 px wide.** Each frame stops at the first tier it fails:

| Tier | Runs on | Computes | Cost per frame |
| --- | --- | --- | --- |
| A | Every sampled frame | Brightness, blur, change from the last frame, a duplicate fingerprint, camera motion for the gyro check | Under a few ms |
| B | Frames that aren't dead (dark, covered, frozen), at 5 per second | MediaPipe hands (21 points each) and faces | A few ms |
| C | Frames with hands, at 1 per second | CLIP embedding, plus clothing and folded-versus-crumpled scores | Tens of ms |

3. **Blank time is found during the pass.** Dead frames never reach tier B, and frames without hands never reach tier C, so the task window builds itself as you go.
4. **Thumbnails saved for Claude** as you go: the first seconds with hands, evenly spaced task frames, and a rolling buffer of the last 3 seconds with hands. The video is never read a second time.
5. **Early exits stop the pass mid-clip:** fingerprints matching an existing clip within the first 3 seconds (duplicate), a confident face (privacy), or the best possible usable time dropping below the contract's minimum.
6. **Decisions from the saved results, with no video:** task window and idle gaps; gyro match, clock offset, the stretched-gyro test for slow motion, and the splice check; the requirement and process rules; the cheap goal check; then the verdict.
7. **Claude only where the cheap checks can't decide:** clips whose goal is unsure, and folding-step labels only for clips the cheap value ranking marks as distinct.
8. **A cost funnel in the report:** clips rejected before decoding, frames read, frames that reached each tier, and Claude calls.

## Plan B: brute force, more likely to work

Same checks and same report, but no gates: every detector runs on every sampled frame, and Claude reviews every clip. The code is simpler, with fewer edge cases, at a higher cost per clip.

- **Separate simple passes:** hands and faces on every frame at 5 per second; CLIP on every frame at 2 per second.
- **Claude on every clip:** goal, folding steps, and process violations, judged from 12 frames.
- **Value uses all three views for every clip,** including Claude's step labels.
- **No early exits:** every clip is read to the end, even once it's clearly rejected.

|  | Plan A | Plan B |
| --- | --- | --- |
| Video reads per clip | 1 | 3 |
| Frames running hands and faces | Only frames that aren't dead | All sampled frames |
| Frames running CLIP | Only frames with hands, 1 per second | All sampled frames, 2 per second |
| Claude calls | Only unsure goals and distinct clips | Every clip |
| Risk of bugs | Higher: gates, buffer, early exits | Lower |
| Fine for 50 clips | Yes | Yes |
| Fine for a lab's monthly volume | Designed for it | Too costly |

## How Plan A and Plan B fit together

Plan B is Plan A with the gates removed, so build B first and add the gates second. B is the safe demo, and A becomes a measured improvement on top of it.

1. **One shared table.** Every detector writes to the same per-frame table, and every check reads only that table. B fills the whole table; A fills only what its gates allow. The checks, report, and value layer never change.
2. **Plan B end to end by about 3:00.** At that point you have a working demo.
3. **Then add A's pieces:** tiers, the thumbnail buffer, early exits, and the cost funnel.
4. **Switch rule:** if A's verdicts don't match B's on the train clips by 3:45, demo B and present A as the design, with whatever savings you measured.
5. **A versus B is a result in itself:** "same verdict on \_\_ of \_\_ clips, with \_\_ times fewer frames run through models and \_\_ versus \_\_ Claude calls." That's the scaling answer for a company handling large volumes of video.

## Build phases

Build TrueTake in parts. Each phase ends with something that works and a number to show for it. Phases 1 and 2 are the Friday hackathon; phases 3–5 come after, and together they match the TrueTake entry on your resume.

| Phase | When | What gets built | Done when |
| --- | --- | --- | --- |
| 1. Core pipeline (Plan B) | Friday | Ingest, the per-frame table, the task contract, gyroscope verification, value ranking, the report | A batch report for all clips |
| 2. Scale and measurement (Plan A) | Friday | Tiers, early exits, the cost funnel, the evaluation tables | A matches B, and the tables are filled |
| 3. Retrieval (RAG) | The week after | A vector database of labeled clips, retrieved as worked examples for Claude | Step-label accuracy with versus without retrieval |
| 4. Cloud Run service | After phase 3 | The pipeline as a deployed API on Google Cloud | A clip uploaded from your phone returns a report link |
| 5. Prompt engineering | Starts after phase 1, then ongoing | Versioned prompt files scored on train clips | The best version chosen before you touch the test clips |

## Phase 3: retrieval layer (RAG)

When Claude labels a new clip's folding steps, show it a few similar clips you've already labeled, as worked examples. Keep it only if it measurably beats no retrieval.

1. **Index.** For each labeled **train** clip, store its CLIP embedding (the task-window frames, averaged) along with its correct steps, its goal verdict, and 2 keyframes. Start with an embedded vector store such as Chroma or LanceDB: one local folder, no server to run.
2. **Retrieve.** Embed the new clip and take its 3 nearest labeled clips.
3. **Prompt.** Put those 3 examples before the new clip's frames. Send each example's keyframes, or just its step list to save tokens, plus its correct steps.
4. **Measure.** On held-out test clips, compare step labels (edit distance to your labels) and goal accuracy with and without retrieval, plus tokens per call. Retrieved images add tokens, so retrieval has to earn its cost.
5. **Leakage rule.** Only train clips go in the index. A test clip that retrieves itself, or its near twin, makes the scores look better than they are.

## Phase 4: deploy on Google Cloud Run

Turn the command-line pipeline into a small web service anyone on a data team can send clips to.

1. **API.** Wrap the pipeline in a FastAPI app in a Docker container: `POST /clips` accepts an upload, and `GET /clips/{id}/report` returns the report.
2. **Storage.** Uploaded videos, per-frame tables, Rerun recordings, and the vector index live in Cloud Storage. The service loads the index from there when it starts.
3. **Secrets.** The Claude API key lives in Secret Manager, never in the container image or the repo.
4. **Batches.** Run large batches as Cloud Run jobs instead of web requests, so a 50-clip batch doesn't hit a request timeout.
5. **Hardware.** Start on CPU. Look at Cloud Run's GPU options for your region only if CLIP turns out to be the bottleneck in your cost funnel.

## Phase 5: prompt engineering

Set up the structure now and tune the wording later. Every Claude call gets its own prompt file, and every change gets scored.

- **Prompts in files, with versions:** `prompts/goal_v1.md`, `prompts/steps_v1.md`. The code loads prompts by name, so rewording a prompt never means changing code.
- **A small scoring script, `prompt_eval.py`:** runs one prompt version over the labeled train clips and reports agreement with your labels, tokens, and cost per clip.
- **Train clips only while tuning.** Run the test clips once, at the end, so the final number is honest.
- **Things to test later:** how many frames and which ones (more near the end for the goal check); rules worded as yes-or-no criteria; a strict JSON schema; asking for evidence frames; with versus without retrieved examples (phase 3); and model tiers, starting with `claude-haiku-5-5` and sending only unsure clips to `claude-sonnet-5-5`.

Starting prompt (v1):

```text
You are reviewing a first-person video of someone doing a task, shown as 8 frames in time order with timestamps.
Task contract: {contract}
Judge only from the frames. Answer each rule with yes, no, or unsure; say unsure rather than guess.
Return only JSON that matches this schema: {schema}
```

## Layer 1: task contract and quality

Each task is a contract with three tiers in priority order: the **goal** (did it happen), the **process** (how it happened), and the **requirements** (format, privacy, setup). Hand-write the folding contract for Friday. Turning a lab's written spec into a contract automatically is a next step.

```yaml
task: fold a T-shirt
goal:            # must be true by the end
  - folded: "garment ends folded into a compact rectangle"
process:         # checked second by second inside the task window
  - hands_visible: "both hands fully in frame while handling the garment"   # 90% of task time
  - garment_visible: "garment in frame"                                     # 90% of task time
  - continuous: "no idle stretch over 3 s"
requirements:
  hard: [landscape, min_1080p, min_30fps, no_slow_motion, no_faces, unedited, not_on_tripod]
  soft: [camera_slightly_down, head_or_chest_mount]
min_usable_seconds: 10
```

**Decision order:** a hard requirement fails → reject. Goal not met → reject. Process gaps → cut those seconds, and accept if the usable time meets the minimum. Soft requirements → warnings. Checks run cheapest first but are decided in this order.

| Check | How it's computed | Default |
| --- | --- | --- |
| Landscape, 1080p+, 30 fps+ | ffprobe width, height, rotation flag, average frame rate | Spec values |
| No slow motion | A capture-rate tag over 60 fps; otherwise the stretched-gyro test in Layer 2 | Any |
| No faces (whole clip, setup included) | MediaPipe faces at 5 per second on frames that aren't dead | Any confident detection |
| Task window | First to last second where hands are in frame and moving for at least 1 second | — |
| Hands fully visible | The contract's number of hands detected, with all 21 MediaPipe points inside the frame and a 2% margin; gaps under 0.5 s filled | 90% of task seconds |
| Garment visible | CLIP clothing score each second | 90% of task seconds |
| Sharp and well lit | Laplacian variance and mean brightness in the task window | Tune on train clips |
| Continuous | Frame change, hand motion, and IMU motion all low | Idle over 3 s |
| Goal | Cheap first: CLIP folded-versus-crumpled on the last 2 seconds against the first 2. Claude judges only the unsure middle, citing frames as evidence. Disagreement goes to human review | — |

Unedited files, camera tilt, and mount are checked in Layer 2. Usable minutes = task window minus flagged seconds. Concrete, timestamped reasons are what real review programs say they needed ([Shaip](https://www.shaip.com/egocentric-video-data-collection-case-study/)).

## Layer 2: verification, gyroscope versus video

The phone's gyroscope is a witness: if the video is real and unedited, how the picture turns must match how the phone turned. One cross-correlation gives both a **match score** (is it real?) and the **clock offset** between the video and sensor files, which labs need anyway to sync their data.

**The calculation, step by step**

1. **Camera motion from video.** For each pair of consecutive frames: convert to grayscale, shrink to 320 px wide, pick about 200 corners (`cv2.goodFeaturesToTrack`), track them into the next frame (`cv2.calcOpticalFlowPyrLK`), and fit a rotation-plus-shift model with outlier rejection (`cv2.estimateAffinePartial2D`, RANSAC). That gives a horizontal shift `dx`, a vertical shift `dy`, and an in-image rotation `dθ`. Divide each by the time between the frames to get three rates: pan, tilt, and roll.
2. **Phone motion from the gyroscope.** The gyro reports turning rates around three axes in rad/s. Interpolate each axis onto the midpoint time of every frame pair.
3. **Clean both.** Smooth with a 0.2-second moving average and subtract each signal's mean.
4. **Match the axes.** Which gyro axis lines up with pan, tilt, or roll depends on how the phone was held. For each video signal, try every gyro axis with both signs and keep the pairing that correlates best. Learn it once per device and orientation on a known-good clip, then lock it.
5. **Cross-correlate.** For each matched pair, compute the correlation r(τ) below at every lag τ from −2 to +2 seconds. The **match score** is the peak value, and the **clock offset** is the lag where it peaks. All three pairs should peak at nearly the same lag (within 2 frames).
6. **Check inside the clip.** Repeat on 5-second windows, sliding 1 second at a time. A window where the match collapses inside an otherwise matching clip marks a splice or edit.
7. **Decide.** If the gyro barely moved (its spread is below a small threshold), report "not enough motion to verify" rather than a failure. Otherwise PASS when the score clears a threshold set on train clips (for example, the level that keeps 95% of real train clips), and also flag offsets over 1 second.

```latex
r(\tau) = \frac{\sum_t (v_t - \bar{v})(g_{t+\tau} - \bar{g})}{\sqrt{\sum_t (v_t - \bar{v})^2}\,\sqrt{\sum_t (g_{t+\tau} - \bar{g})^2}}
```

Here v is a video rate signal and g is the matched gyro axis. Correlation ignores scale, so you never need the camera's focal length to turn pixels into radians.

**What it should catch**

| Attack | Expected result |
| --- | --- |
| Sensor file from a different clip | Low match score |
| Sensor file with its clock shifted | Match found, but at the wrong offset |
| Screen re-filmed from a tripod | Video shows motion, gyro is flat: low match |
| Spliced or inserted segment | Match collapses in that window |
| Stock or generated video with made-up sensor data | Low match score |
| Screen re-filmed with a hand-held phone | Partial match, because the phone really moved. Known gap |

**Limits and cost**

- Close-up hands and cloth add motion that isn't camera rotation, which lowers scores at tabletop distance. Set the threshold from real folding clips, not walking clips.
- Phone video stabilization removes the shake you're trying to match. Turn it off in the recording app, and test this first.
- Heavy motion blur loses tracked corners. Those frames count as inconclusive.
- **Cost:** tracking 200 points at 320 px is light, so a 30-second clip should take seconds on a laptop. Measure it and put the number in the demo. The hardware is free: every modern smartphone has a gyroscope.

## Other verification checks and device support

Most checks need only the video, so TrueTake works with any camera. Sensor-based checks run when a sensor file exists and are marked "unverified" when it doesn't.

| Check | Catches | How | Needs sensors? |
| --- | --- | --- | --- |
| Duplicates | Re-uploads, and trimmed, mirrored, or re-encoded copies | Perceptual hash of sampled frames and their mirror images (`imagehash`), plus CLIP similarity across the batch | No |
| Frame timing | Edits and dropped frames | Jumps or gaps in frame timestamps | No |
| File metadata | Re-exported or edited files, wrong format | `ffprobe` encoder and creation tags, rotation, frame rate | No |
| Camera tilt | "Angled slightly downward" rule | Direction of gravity in the accelerometer gives the pitch angle | Yes; marked unknown without sensors |
| Mount | Phone held by hand or set on a surface | Near-zero motion means it sat on a surface; hand tremor versus head motion patterns (stretch) | Yes |
| Gyroscope vs video | Borrowed, shifted, spliced, or re-filmed footage | Previous section | Yes |

**Device modes**

- **iPhone or Android with a logging app:** every check runs. Pick one app that exists on both platforms and records video plus gyroscope and accelerometer (test Sensor Logger first). In `ingest.py`, convert both platforms' exports to the same columns and units: seconds, rad/s, m/s². Android and iPhone apps may differ in units, for example degrees versus radians per second.
- **Any camera with no sensor file** (webcam, older phone, missing file): quality and value run normally. Verification shows "unverified: no motion data" instead of a rejection, and the lab decides its own policy.
- **Action cameras:** many, including GoPro, store gyroscope data inside the video file, and stabilization tools like Gyroflow read it. Reading that data would bring these cameras into full verification. That's a stretch goal.

## Layer 3: value

A clip's value is how much new behavior it adds to the clips already kept. The 40th standard fold adds almost nothing, while the only fold in thirds, or a dropped sleeve and its fix, adds a lot. Failure-and-recovery cases are a known gap in robot data ([Unidata](https://www.unidata.pro/blog/egocentric-data-collection-for-robot-training/)), and a robot trained on one demonstrator's style picks up that person's quirks ([DDD](https://www.digitaldividedata.com/?p=24288)).

Only clips that pass quality are scored, so "unusual" can't just mean "broken."

**Three views of each clip, each with a distance**

1. **Scene (what and where):** CLIP ViT-B-32 embeddings of 8 frames, averaged. Distance = 1 − cosine similarity. This catches same shirt, same table, same lighting.
2. **Hand motion (how the hands move):** MediaPipe Hands gives the wrist, thumb tip, and index tip of each hand per frame. Normalize to the frame size and resample to 64 time steps. Distance = dynamic time warping (DTW), about 15 lines of numpy.
3. **Strategy (which fold):** Claude gets 8 ordered frames and a fixed step list: smooth, sleeves-in, fold-half-long, fold-half-short, fold-thirds, roll, flip, drop, recover, other. It returns the step sequence plus tags for garment, surface, and leading hand. Distance = edit distance divided by the longer sequence's length.

**Combining and ranking**

1. Divide each view's distances by that view's median, then average the three into one distance between every pair of clips.
2. **Value ranking:** start from the most typical clip (smallest average distance to the rest). Repeatedly add the clip that is farthest from everything already chosen. Each clip's value is its distance to the nearest chosen clip at the moment it's added, so early picks are the most valuable.
3. **Redundancy:** a clip whose nearest neighbor is closer than a threshold is marked redundant, and the report names its twin.
4. **Film next:** the spec file lists the wanted variations (garments × surfaces × styles × leading hand). Count the accepted clips per combination using Claude's tags; empty combinations become the film-next list.
5. **Map:** lay out the distance matrix in 2D (scikit-learn MDS), colored by Claude's strategy label. Clicking a clip shows why it ranked where it did.

Diversity selection is an established field. What's new here is scoring behavior in robot footage and giving reasons a reviewer can read.

## Data to record with friends

About 57 clips with 3–4 people takes roughly an hour. Prepared data is usually allowed when code isn't, but confirm with the organizers and mention it in the pitch.

| Group | Clips | How to make them |
| --- | --- | --- |
| Standard fold | 20 | Each person folds T-shirts their usual way. Make this the biggest group so redundancy is real |
| Other styles | 12 | Fold in thirds, roll, fold on a bed or lap, one-handed, a towel, a hoodie |
| Mistake and recovery | 4 | Drop a sleeve or misfold, then fix it |
| Goal not met | 5 | Stop halfway, fold then shake it out, or walk away mid-task |
| Quality failures | 8 | Too dark, hands out of frame, idle pause over 3 seconds, vertical video |
| Recorded attacks | 4 | Phone on a tripod filming a laptop that plays a good clip |
| Video only | 4 | Recorded without the sensor app, or on another camera |

**Attacks made in code** (no filming): pair a video with another clip's sensor file, shift a sensor file's clock by 0.5–3 seconds, and splice two clips together. Build these from train clips only.

**Recording rules**

- Same chest or head mount for everyone, angled slightly down, landscape, video stabilization off.
- Faces out of frame, and consent from everyone who records.
- Note who folded and how as you go. Those notes become your hidden labels.

**Labels and split**

```csv
clip_id,person,style,garment,surface,goal_met,quality,attack,split
clip_001,A,standard,tshirt,table,yes,good,none,train
clip_031,C,thirds,tshirt,table,yes,good,none,test
clip_047,B,standard,tshirt,table,yes,dark,none,test
clip_052,D,standard,tshirt,table,no,good,none,test
```

Hold out about a third as the test set, with every style present in both train and test. Never tune a threshold on test clips.

## Setup checklist (before Friday, no project code)

The riskiest item is the logging app, so test it first.

- [ ] **Logging app (Tuesday):** on your iPhone, record 20 seconds of video with gyroscope and accelerometer. Export it and confirm you can read the timestamps, units, and sample rate. Confirm video stabilization can be turned off. If a friend has an Android, test the same app there.
- [ ] Ask the organizers: is pre-recorded data allowed, and what's the demo time and format?
- [ ] Python 3.11 with uv: `uv venv`, then `uv pip install opencv-python mediapipe open_clip_torch torch scikit-learn pandas numpy rerun-sdk anthropic imagehash matplotlib pyyaml`
- [ ] ffmpeg installed, so `ffprobe` works (on a Mac, `brew install ffmpeg`).
- [ ] Rerun opens: `python -c "import rerun as rr; rr.init('test', spawn=True); rr.log('hello', rr.TextLog('ok'))"`
- [ ] Claude API key created at platform.claude.com → Settings → API keys, saved as the `ANTHROPIC_API_KEY` environment variable (never committed to Git), billing set up, and one test call that sends an image works.
- [ ] CLIP weights downloaded once on good Wi-Fi.
- [ ] Clips recorded, labeled, and split (see the data section).
- [ ] Empty GitHub repo, charger, extension cord.
- [ ] MiniMed call time confirmed (see the timeline).

## Day-of timeline (Friday, Nov 6)

Feature freeze at 4:15. Each block has a test for done and a fallback if it runs long.

| Time | Build | Done when | If behind |
| --- | --- | --- | --- |
| 11:00–11:30 | Setup, repo, paste the brief into Claude | Repo pushed | — |
| 11:30–12:30 | `ingest.py` (video, sensors, metadata), `viz_rerun.py` | One clip plays in Rerun with gyro plots under it | Skip the accelerometer |
| 12:30–1:30 | `detect.py` (Plan B: every detector fills the per-frame table), `quality.py` with the task contract | Usable minutes and flags for 3 clips | Drop the face check |
| 1:30–2:15 | `verify_gyro.py`, `duplicates.py`, `make_attacks.py` | A real clip passes; a swapped-sensor clip fails | Drop the 5-second window check |
| 2:15–3:00 | `value.py`, Claude goal and steps on every clip | Ranking and map for all clips. **Plan B is complete** | Drop the hand-motion view |
| 3:00–3:45 | Plan A: tiers, thumbnail buffer, early exits, cost funnel | A's verdicts match B's on train clips | Demo Plan B; present A as the design |
| 3:45–4:15 | `evaluate.py`, `report.py` | All result tables filled, including A versus B | Report whatever finished |
| 4:15–5:15 | Slides, backup video, rehearse twice | 3-minute demo runs clean | Cut slides, keep the demo |
| 5:15–6:00 | Demos | — | — |

The demo time isn't posted yet, so ask at check-in and shift the last two rows. If Or Baron confirmed 1:00 PM PST, that's 4:00 PM ET. In that case, skip Plan A if it isn't done by 3:15, run `evaluate.py` from 3:15 to 4:00, freeze at 4:00, and take the 20-minute call somewhere quiet.

## Building with Claude

One file per prompt, run it on a real clip, read the code, then commit. Mercor's own engineering interview includes an AI-assisted build where you must stand behind every line ([source](https://www.techinterview.org/post/3233477136/mercor-interview-guide/)), so expect judges to ask about any file.

**Workflow for every file**

1. Start each new session by pasting the brief below.
2. Give the function name, its inputs, and what it returns.
3. Run it on one good clip and one bad clip, and look at the Rerun view.
4. If you can't explain a line, ask Claude to explain or simplify it.
5. Commit after every working file.
6. Keep a short decision log (thresholds, axis mapping, why each view) for the Q&A.

**Brief (paste at the start of each session)**

```text
I'm building TrueTake at a 7-hour hackathon: a Python CLI that reviews first-person clips of people folding clothes, collected to train robots. Three layers: quality (does the clip meet the spec), verification (does the video's motion match the phone's gyroscope), value (does the clip add a new folding behavior). Output: a batch report, per-clip cards with timestamped reasons, Rerun views, and a 2D map of clips.

Rules: Python 3.11. Small files, plain functions, readable code; I must explain every line to judges. All times are float seconds from the start of the clip. Sensor units: rad/s and m/s^2. Thresholds are named constants at the top of each file. A clip may have no sensor file; then imu is None and sensor checks return passed=None with reason "no motion data".

@dataclass
class Recording:
    clip_id: str
    video_path: str
    frame_times: np.ndarray        # seconds, one per decoded frame
    meta: dict                     # ffprobe fields: width, height, fps, rotation, tags
    imu: pd.DataFrame | None       # columns t, gx, gy, gz, ax, ay, az

@dataclass
class CheckResult:
    name: str
    passed: bool | None            # None = inconclusive or unverified
    score: float
    reason: str
    flags: list[tuple[float, float, str]]   # (start_s, end_s, note)
```

**Prompts, in build order**

1. **ingest.py:** "Write `load_recording(video_path, sensors_path_or_None) -> Recording`. Here are the first 5 lines of my sensor export: \[paste\]. Convert timestamps to seconds and gyro to rad/s. Read frame times with OpenCV and metadata with ffprobe. Print duration, frame count, gyro rate."
2. **viz\_rerun.py:** "Write `show(rec, results)`: every third frame as an image on a seconds timeline, gyro and accelerometer as scalar plots, each flag as a text log at its start time."
3. **quality.py:** "Implement the Layer 1 table: \[paste the table\]. Score each second and return CheckResults plus usable seconds."
4. **verify\_gyro.py:** "Implement steps 1–7 of the gyroscope section exactly: \[paste the steps\]. Also return the per-window match and the clock offset, and log the video rate and matched gyro axis to Rerun so I can see them line up."
5. **duplicates.py:** "Perceptual hashes of 8 frames per clip and their mirror images. Report pairs with a near-identical hash sequence, and pairs with CLIP cosine similarity above a threshold."
6. **make\_attacks.py:** "From train clips, write swapped-sensor pairs, clock-shifted sensor files (0.5–3 s), and two-clip splices, with labels."
7. **views.py:** "Three functions: CLIP scene vector, MediaPipe hand paths resampled to 64 steps, and Claude strategy steps from this fixed list: \[paste\]. Cache every result to disk."
8. **value.py:** "Build the combined distance matrix, run farthest-point ranking, mark redundant pairs, fill the coverage plan from tags, and draw the MDS map colored by strategy."
9. **report.py and evaluate.py:** "Print the batch report in this format: \[paste the example\]. evaluate.py fills the three result tables in the Evaluation section using labels.csv, test split only."

**Mistakes to watch for**

- Mixed timestamp units (ns, ms, s), and degrees versus radians. Print the duration and the gyro range.
- OpenCV gives BGR color; MediaPipe and CLIP expect RGB.
- Rotated phone video: check one frame by eye.
- Video stabilization left on, which flattens the video's motion.
- Wrong axis or sign in the gyro mapping. Plot both signals in Rerun before trusting a score.
- Leakage: split by clip, and never tune a threshold on test clips.
- MediaPipe API changes between versions; paste the error and your installed version.
- Claude's JSON may arrive wrapped in extra text; parse it defensively and cache responses.

## Evaluation: four measured results

Every number comes from test clips the system never trained or tuned on, scored against your hidden labels.

1. **Value beats random.** Ask for the 10 most valuable clips and count how many of your folding styles they cover. Compare with the average over 1,000 random draws of 10 clips. Also check two things: what share of standard-fold clips get marked redundant, and whether the recovery clips rank in the top half.
2. **Verification catches attacks.** Compare match scores for real test clips against each attack type. Report attacks caught, real clips wrongly flagged, and inconclusive clips.
3. **Quality and goal checks agree with you.** Per check, how often its verdict matches your label, including the goal-not-met clips. Compare the cheap goal check alone, Claude alone, and the cheap-first combination, and report what share of clips the combination sent to Claude.
4. **Plan A matches Plan B for less.** Run both on the same clips. Report how often their verdicts agree, and the frames and Claude calls each one used.

| Measure | Result |
| --- | --- |
| Styles covered by the top 10 picks | \_\_ of \_\_ (random 10: \_\_ on average) |
| Standard-fold clips marked redundant | \_\_ of \_\_ |
| Attacks caught | \_\_ of \_\_ |
| Real clips wrongly flagged | \_\_ of \_\_ |
| Quality checks agreeing with your labels | \_\_% |
| Goal verdicts agreeing with your labels (cheap / Claude / combined) | \_\_% / \_\_% / \_\_% |
| Clips the combined goal check sent to Claude | \_\_ of \_\_ |
| Usable minutes kept versus whole-clip rejection | \_\_ versus \_\_ |
| Plan A verdicts matching Plan B | \_\_ of \_\_ clips |
| Frames run through models (A / B) | \_\_ / \_\_ |
| Claude calls (A / B) | \_\_ / \_\_ |

With about 50 clips, these numbers are small-sample evidence. Say that, and say what you'd expect at 500. Higher coverage is a stand-in for better robot training, not proof. Proving it takes a training run, which is your stated next step.

## Demo script (3 minutes)

Open with the problem, show the map and the gyro overlay live, and close on the numbers.

1. **0:00–0:20, the problem.** Labs pay for approved hours of first-person footage, and Mercor itself recruits people to film chores. Most of those hours repeat the same motions.
2. **0:20–1:00, the batch report.** 52 clips in: usable minutes, rejections with timestamped reasons, verification results, and the film-next list.
3. **1:00–1:40, the map.** Show the large cluster of standard folds, then click the lone fold-in-thirds clip and the sleeve-drop recovery, each with its reason.
4. **1:40–2:10, verification live.** In Rerun, the video's rotation and the gyro lock together on a real clip. Swap in another clip's sensor file: they fall apart and the verdict flips to REJECT.
5. **2:10–2:40, results.** Top-10 style coverage versus random, and attacks caught.
6. **2:40–3:00, where it goes.** The same pipeline fits workplace footage. Mercor already pays businesses for wearable-camera video of their teams at work ([Mercor](https://www.mercor.com/robotics/)). After that comes teleoperation data, where each hour costs the most.

**Backup plan:** a screen recording of the whole demo on the laptop, plus saved outputs for every clip. Run nothing live that takes longer than 10 seconds.

## Likely judge questions

Short answers you can say without notes.

| Question | Answer |
| --- | --- |
| What if the camera has no gyroscope? | Quality and value still run. Verification says "unverified" instead of rejecting, and the lab sets the policy. Many action cameras store gyro data in the file, which is the next device to support. |
| Couldn't someone fake the gyro data? | They'd have to generate sensor data that agrees with the motion in every frame, which is far harder than filming. The window check also catches splices. It raises the cost of cheating; it isn't proof. |
| Isn't an unusual clip often just a bad one? | Only clips that pass quality get scored for value, and the strategy steps say why a clip differs. A human approves the top picks. |
| Is diverse data really better for robots? | Practitioners report that diversity matters more than volume, but coverage is a stand-in. A training run comparing picked versus random clips is the next step. |
| How is this different from current review? | Programs already have AI and humans accept or reject each clip ([MIT Technology Review](https://www.technologyreview.com/2026/04/01/1134863/humanoid-data-training-gig-economy-2026-breakthrough-technology/amp/)). TrueTake adds per-second usable minutes, sensor verification, and value scoring. |
| Why not let Claude judge everything? | Cost and consistency. Cheap checks run on every clip, and Claude runs only on clips that pass them. |
| What did Claude write, and what did you? | Claude wrote most of the code from my specs. I designed the three layers, the gyro method, the splits, and the thresholds, and I can walk through any file. |
| What about privacy? | Footage shows homes. Face checks flag the seconds to blur, and raw video should be kept only as long as review needs. |

## Stretch goals and cut list

Never cut `evaluate.py`. The measured tables matter more than any extra feature.

**Stretch, in order, only after the freeze checklist is met:**

1. Run 50 public factory clips (for example from Egocentric-1M) through the pipeline unchanged, to show it generalizes.
2. A screen re-film classifier (CLIP embeddings plus logistic regression) to close the hand-held re-film gap.
3. Mount detection from IMU patterns: head, chest, hand-held, or resting on a surface.
4. Read gyroscope data embedded in action-camera files.
5. Turn film-next gaps into priced bounties.

**Cut first if behind:** the 5-second window check, then the hand-motion view, then the face check, then mirrored-duplicate detection.

## Sources

- [Mercor: Household Activity Video Contributor (US based)](https://hackajob.com/job/dc5bf67f-760d-11f1-a7b8-0a05e249917d-household-activity-video-contributor-us-based-train-ai-models-part-time)
- [Mercor: Software Engineer, Robotics Data](https://wellfound.com/jobs/4523083-software-engineer-robotics-data)
- [Mercor: wearable-camera footage from businesses](https://www.mercor.com/robotics/)
- [Public egocentric collection spec and rejection rules](https://www.workana.com/en/job/egocentric-video-data-collection)
- [Shaip: 30,000-hour egocentric collection case study](https://www.shaip.com/egocentric-video-data-collection-case-study/)
- [Encord: diversity in egocentric robot data](https://encord.com/blog/egocentric-data-collection-for-robotics/)
- [DDD: coverage matrices and demonstrator diversity](https://www.digitaldividedata.com/?p=24288)
- [Unidata: what works in egocentric robot data](https://www.unidata.pro/blog/egocentric-data-collection-for-robot-training/)
- [MIT Technology Review: gig workers training humanoids](https://www.technologyreview.com/2026/04/01/1134863/humanoid-data-training-gig-economy-2026-breakthrough-technology/amp/)
- [Robot data gold rush (Micro1 volumes)](https://ai2.work/blog/the-robot-data-gold-rush-filming-human-labor-to-train-machines)
- [How Mercor interviews engineers](https://www.techinterview.org/post/3233477136/mercor-interview-guide/)
