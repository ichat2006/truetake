"""Load video timing and an optional, already-normalized sensor CSV."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from truetake.models import Recording


FFPROBE_TIMEOUT_SECONDS = 120
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
SENSOR_COLUMNS = ["t", "gx", "gy", "gz", "ax", "ay", "az"]


def load_sensors(sensors_path: str) -> pd.DataFrame:
    """Read seconds relative to video start, rad/s, and m/s^2 without guessing units."""
    path = Path(sensors_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Sensors must be a normalized .csv file.")
    if not path.is_file():
        raise FileNotFoundError(f"Sensor file not found: {path}")

    table = pd.read_csv(path)
    missing = [column for column in SENSOR_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Sensor CSV is missing columns: {', '.join(missing)}")
    if table.empty:
        raise ValueError("Sensor CSV contains no samples.")

    try:
        table = table[SENSOR_COLUMNS].astype("float64")
    except (ValueError, TypeError) as error:
        raise ValueError("Sensor columns must contain only numbers.") from error
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Sensor columns must contain finite values, with no empty cells.")
    if len(table) > 1 and not np.all(np.diff(table["t"]) > 0):
        raise ValueError("Sensor timestamps must be strictly increasing, without duplicates.")
    return table


def load_recording(video_path: str, sensors_path: str | None = None) -> Recording:
    """Load presentation times with ffprobe; never derive them from nominal FPS."""
    path = Path(video_path)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Unsupported video extension; use MP4, MOV, M4V, MKV, AVI, or WebM.")
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration:stream_tags:stream_side_data:"
        "frame=best_effort_timestamp_time,pkt_duration_time",
        "-of", "json", str(path.resolve()),
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required; install ffmpeg before loading video.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("ffprobe timed out while reading video timestamps.") from error
    except subprocess.CalledProcessError as error:
        raise ValueError("ffprobe could not decode the video.") from error

    try:
        probe = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("ffprobe returned invalid JSON.") from error
    streams = probe.get("streams", [])
    frames = probe.get("frames", [])
    if not streams or not frames:
        raise ValueError("Video has no decodable frames.")
    stream = streams[0]

    try:
        source_times = np.array(
            [float(frame["best_effort_timestamp_time"]) for frame in frames],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every video frame must have a presentation timestamp.") from error
    if not np.isfinite(source_times).all():
        raise ValueError("Video timestamps must be finite.")
    if len(source_times) > 1 and not np.all(np.diff(source_times) > 0):
        raise ValueError("Video presentation timestamps must be strictly increasing.")
    frame_times = source_times - source_times[0]

    try:
        width = int(stream["width"])
        height = int(stream["height"])
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Video dimensions or frame rate are invalid.") from error
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps < 0:
        raise ValueError("Video dimensions or frame rate are invalid.")

    # Retain source rotation; later image decoding must apply it exactly once.
    tags = stream.get("tags", {})
    rotation = float(tags.get("rotate", 0))
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = float(side_data["rotation"])
    if not np.isfinite(rotation) or rotation % 90 != 0:
        raise ValueError("Only right-angle video rotation is supported.")
    rotation = int(rotation) % 360
    display_width, display_height = width, height
    if rotation in (90, 270):
        display_width, display_height = height, width

    # Prefer the last frame's actual duration; report when an estimate was needed.
    last_duration = float(frames[-1].get("pkt_duration_time", 0))
    duration_estimated = not np.isfinite(last_duration) or last_duration <= 0
    if duration_estimated:
        if len(frame_times) > 1:
            last_duration = float(np.median(np.diff(frame_times)))
        elif fps > 0:
            last_duration = 1 / fps
        else:
            raise ValueError("Cannot determine the duration of a single-frame video.")

    meta = {
        "width": width,
        "height": height,
        "display_width": display_width,
        "display_height": display_height,
        "fps": fps,
        "rotation": rotation,
        "tags": tags,
        "duration_s": float(frame_times[-1] + last_duration),
        "duration_estimated": duration_estimated,
    }
    imu = None if sensors_path is None else load_sensors(sensors_path)
    return Recording(path.stem, str(path), frame_times, meta, imu)
