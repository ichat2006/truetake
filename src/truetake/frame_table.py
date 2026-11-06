"""Build the shared table using actual source-frame presentation times."""

import numpy as np
import pandas as pd

from truetake.models import Recording


SAMPLE_HZ = 10.0
SAMPLE_TIME_TOLERANCE_S = 0.000001
MAX_SENSOR_GAP_S = 0.1
SENSOR_AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
GYRO_SMOOTHING_S = 0.2


def build_table(rec: Recording, contract: dict | None = None) -> pd.DataFrame:
    """Select distinct frames at/after 10 Hz targets; never invent missing frames."""
    times = rec.frame_times
    duration = rec.meta["duration_s"]
    if (len(times) == 0 or not np.isfinite(times).all() or times[0] != 0
            or np.any(np.diff(times) <= 0) or not np.isfinite(duration)
            or duration <= times[-1]):
        raise ValueError("Recording must have increasing frame times from zero and a valid duration.")
    targets = np.arange(0, duration, 1 / SAMPLE_HZ)
    indices = np.searchsorted(times, targets - SAMPLE_TIME_TOLERANCE_S)
    indices = np.unique(indices[indices < len(times)])
    selected = times[indices]
    table = pd.DataFrame({
        "source_frame": indices,
        "t": selected,
        "sample_end": np.append(selected[1:], duration),
        "sensor_present": rec.imu is not None,
    })
    table["clip_meta"] = pd.Series([rec.meta.copy()] + [None] * (len(table) - 1), dtype=object)
    table["contract"] = pd.Series([contract] + [None] * (len(table) - 1), dtype=object)
    for axis in SENSOR_AXES:
        table[axis] = np.nan
    for axis in ("gx", "gy", "gz"):
        table[f"gyro_{axis}"] = np.nan
    table["imu_valid"] = False
    if rec.imu is None:
        return table

    # Preserve missing support and long gaps; never extrapolate sensor values.
    sensor_times = rec.imu["t"].to_numpy()
    right = np.searchsorted(sensor_times, selected)
    right = np.minimum(right, len(sensor_times) - 1)
    left = np.maximum(right - 1, 0)
    exact = sensor_times[right] == selected
    valid = ((selected >= sensor_times[0]) & (selected <= sensor_times[-1])
             & (exact | (sensor_times[right] - sensor_times[left] <= MAX_SENSOR_GAP_S)))
    for axis in SENSOR_AXES:
        table.loc[valid, axis] = np.interp(selected[valid], sensor_times, rec.imu[axis])
    # Smooth at the original sensor rate before reducing to video-table times.
    segments = np.split(np.arange(len(sensor_times)),
                        np.flatnonzero(np.diff(sensor_times) > MAX_SENSOR_GAP_S) + 1)
    for axis in ("gx", "gy", "gz"):
        smoothed = np.empty(len(sensor_times))
        for segment in segments:
            signal = pd.Series(rec.imu[axis].to_numpy()[segment],
                               index=pd.to_timedelta(sensor_times[segment], unit="s"))
            smoothed[segment] = signal.rolling(pd.Timedelta(seconds=GYRO_SMOOTHING_S),
                                              center=True, closed="both").mean()
        table.loc[valid, f"gyro_{axis}"] = np.interp(selected[valid], sensor_times, smoothed)
    table["imu_valid"] = valid
    return table
