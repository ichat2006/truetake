"""Shared records and check results used throughout TrueTake."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Recording:
    clip_id: str
    video_path: str
    frame_times: np.ndarray  # Float seconds from the first presented video frame.
    meta: dict
    imu: pd.DataFrame | None  # t, gx, gy, gz, ax, ay, az; seconds, rad/s, m/s^2.


@dataclass
class CheckResult:
    name: str
    passed: bool | None
    score: float
    reason: str
    flags: list[tuple[float, float, str]]
