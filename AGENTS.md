# TrueTake

Python CLI that reviews first-person clips of people doing tasks (first task: folding clothes) collected to train robots. Three layers: quality (task contract: goal, process, requirements), verification (video motion vs gyroscope), value (does the clip add a new behavior). Output: batch report, per-clip cards with timestamped reasons, Rerun views, a 2D clip map. Runtime AI calls use the Claude API (anthropic SDK).

## Commands
- Setup: `uv sync`
- Tests: `uv run pytest -q`
- Run one clip: `uv run python cli.py data/clips/clip_001.mp4 --sensors data/clips/clip_001_sensors.csv --contract contracts/fold.yaml`

## Rules
- One file per task. Run the tests and the command above on 2 clips before saying a task is done.
- Python 3.11, plain functions, readable code. I must explain every line to judges.
- All times are float seconds from clip start. Sensor units: rad/s, m/s^2.
- Thresholds are named constants at the top of each file.
- Every detector writes to the per-frame table; every check reads only that table.
- A clip may have no sensor file: imu is None, and sensor checks return passed=None with reason "no motion data".
- The laptop has no GPU: run torch on CPU, batch CLIP frames, and keep frame sizes small.
- Never read, print, or commit `.env` or API keys.
- Never modify `data/` or tune anything on clips with split=test in labels.csv.
- Ask before adding a dependency.

## Core types
```python
@dataclass
class Recording:
    clip_id: str
    video_path: str
    frame_times: np.ndarray        # seconds
    meta: dict                     # ffprobe: width, height, fps, rotation, tags
    imu: pd.DataFrame | None       # t, gx, gy, gz, ax, ay, az

@dataclass
class CheckResult:
    name: str
    passed: bool | None            # None = inconclusive or unverified
    score: float
    reason: str
    flags: list[tuple[float, float, str]]   # (start_s, end_s, note)
```

## Design
Full design: docs/PLAN.md. Before each task, read only the section it names.
