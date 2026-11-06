"""Estimate camera motion from sampled images and store it in the shared table."""

import cv2
import numpy as np
import pandas as pd

from truetake.detect_basic import iter_sampled_frames
from truetake.models import Recording


MAX_CORNERS = 200
CORNER_QUALITY = 0.01
MIN_CORNER_DISTANCE_PX = 7
MIN_TRACKED_POINTS = 20
LK_WINDOW_PX = 21
LK_PYRAMID_LEVELS = 3
RANSAC_ERROR_PX = 2.0
MIN_INLIER_FRACTION = 0.5
MAX_PAIR_INTERVAL_S = 0.2


def measure_pair(previous: np.ndarray, current: np.ndarray, dt: float) -> dict:
    """Measure center translation in pixels/s and image rotation in rad/s."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Frame interval must be positive and finite.")
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("Flow requires two grayscale images of the same size.")
    result = {"flow_pan": None, "flow_tilt": None, "flow_roll": None,
              "flow_inlier_fraction": None, "flow_points": 0, "flow_status": "frame_gap"}
    if dt > MAX_PAIR_INTERVAL_S:
        return result
    points = cv2.goodFeaturesToTrack(previous, maxCorners=MAX_CORNERS,
                                    qualityLevel=CORNER_QUALITY,
                                    minDistance=MIN_CORNER_DISTANCE_PX)
    if points is None or len(points) < MIN_TRACKED_POINTS:
        result["flow_status"] = "insufficient_features"
        return result
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None,
        winSize=(LK_WINDOW_PX, LK_WINDOW_PX), maxLevel=LK_PYRAMID_LEVELS,
    )
    if tracked is None or status is None:
        result["flow_status"] = "tracking_failed"
        return result
    source, target = points.reshape(-1, 2), tracked.reshape(-1, 2)
    height, width = current.shape
    valid = ((status.ravel() == 1) & np.isfinite(target).all(axis=1)
             & (target[:, 0] >= 0) & (target[:, 0] < width)
             & (target[:, 1] >= 0) & (target[:, 1] < height))
    result["flow_points"] = int(valid.sum())
    if result["flow_points"] < MIN_TRACKED_POINTS:
        result["flow_status"] = "insufficient_tracks"
        return result
    matrix, inliers = cv2.estimateAffinePartial2D(
        source[valid], target[valid], method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_ERROR_PX,
    )
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        result["flow_status"] = "fit_failed"
        return result
    fraction = float(inliers.mean())
    result["flow_inlier_fraction"] = fraction
    if fraction < MIN_INLIER_FRACTION:
        result["flow_status"] = "low_inliers"
        return result
    # Measure translation at image center so pure roll does not create fake pan.
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    shift = matrix[:, :2] @ center + matrix[:, 2] - center
    result.update(flow_pan=float(shift[0] / dt), flow_tilt=float(shift[1] / dt),
                  flow_roll=float(np.arctan2(matrix[1, 0], matrix[0, 0]) / dt),
                  flow_status="ok")
    return result


def detect_flow(rec: Recording, table: pd.DataFrame) -> pd.DataFrame:
    """Store each pair's actual midpoint time, interval, motion, and fit status."""
    table = table.copy()
    for column in ("flow_t", "flow_dt", "flow_pan", "flow_tilt", "flow_roll", "flow_inlier_fraction"):
        table[column] = np.nan
    table["flow_points"] = 0
    table["flow_status"] = "no_previous"
    previous = None
    previous_time = None
    for row, frame in iter_sampled_frames(rec, table):
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time = float(table.at[row, "t"])
        if previous is not None:
            dt = time - previous_time
            table.at[row, "flow_t"] = (time + previous_time) / 2
            table.at[row, "flow_dt"] = dt
            for column, value in measure_pair(previous, current, dt).items():
                table.at[row, column] = value
        previous, previous_time = current, time
    return table
