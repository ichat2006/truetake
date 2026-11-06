"""Measure every sampled frame without quality gates or early rejection."""

from collections.abc import Iterator

import cv2
import numpy as np
import pandas as pd

from truetake.models import Recording


ANALYSIS_WIDTH = 320
GRAYSCALE_MAX = 255.0


def iter_sampled_frames(rec: Recording, table: pd.DataFrame,
                        max_width: int = ANALYSIS_WIDTH) -> Iterator[tuple[int, np.ndarray]]:
    """Yield upright BGR images; detectors default to the small analysis size."""
    capture = cv2.VideoCapture(rec.video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open the video.")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rows = dict(zip(table["source_frame"], table.index))
    rotation = rec.meta["rotation"]
    rotations = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_CLOCKWISE}
    try:
        for source_frame in range(len(rec.frame_times)):
            if not capture.grab():
                raise ValueError(f"Video decoding stopped at source frame {source_frame}.")
            if source_frame not in rows:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise ValueError(f"Could not decode source frame {source_frame}.")
            if rotation:
                frame = cv2.rotate(frame, rotations[rotation])
            height, width = frame.shape[:2]
            if width > max_width:
                frame = cv2.resize(frame, (max_width, max(1, round(height * max_width / width))),
                                   interpolation=cv2.INTER_AREA)
            yield rows[source_frame], frame
    finally:
        capture.release()


def detect_basic(rec: Recording, table: pd.DataFrame) -> pd.DataFrame:
    """Write brightness [0,255], Laplacian variance, and normalized frame change."""
    table = table.copy()
    for column in ("brightness", "blur_var", "frame_change"):
        table[column] = np.nan
    previous = None
    for row, frame in iter_sampled_frames(rec, table):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        table.at[row, "brightness"] = float(gray.mean())
        table.at[row, "blur_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if previous is not None:
            table.at[row, "frame_change"] = float(cv2.absdiff(gray, previous).mean() / GRAYSCALE_MAX)
        previous = gray
    return table
