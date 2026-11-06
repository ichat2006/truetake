"""CPU hand/face measurements at 5 Hz, including dark and inactive frames."""

from pathlib import Path
import ctypes
from contextlib import contextmanager

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from truetake.detect_basic import iter_sampled_frames
from truetake.models import Recording


PEOPLE_HZ = 5.0
LANDMARK_MARGIN = 0.02
DETECTION_CONFIDENCE = 0.5
REQUIRED_HANDS = 2


def summarize_hands(result) -> list[dict]:
    return [{"label": categories[0].category_name, "confidence": float(categories[0].score),
             "points": [[float(p.x), float(p.y), float(p.z)] for p in points]}
            for points, categories in zip(result.hand_landmarks, result.handedness)]


def hands_inside(hands: list[dict]) -> bool:
    return len(hands) >= REQUIRED_HANDS and all(
        len(hand["points"]) == 21 and all(LANDMARK_MARGIN <= p[0] <= 1 - LANDMARK_MARGIN
                                        and LANDMARK_MARGIN <= p[1] <= 1 - LANDMARK_MARGIN
                                        for p in hand["points"])
        for hand in hands)


def people_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["people_status"] = "not_sampled"
    for column in ("hand_count", "hands_inside", "hand_motion", "face_confidence"):
        table[column] = np.nan
    for column in ("hand_landmarks", "face_boxes"):
        table[column] = pd.Series([None] * len(table), index=table.index, dtype=object)
    return table


@contextmanager
def people_models(model_dir: str):
    model_dir = Path(model_dir)
    local_gles = model_dir / "runtime/usr/lib/x86_64-linux-gnu/libGLESv2.so.2"
    if local_gles.exists():
        ctypes.CDLL(str(local_gles.resolve()), mode=ctypes.RTLD_GLOBAL)
    paths = [model_dir / "hand_landmarker.task", model_dir / "face_detector.tflite"]
    if not all(path.is_file() for path in paths):
        raise ValueError("Download the hand and face model files into outputs/models first.")
    base = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    hand_options = vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(paths[0]), delegate=base.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_hands=REQUIRED_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=DETECTION_CONFIDENCE)
    face_options = vision.FaceDetectorOptions(
        base_options=base(model_asset_path=str(paths[1]), delegate=base.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, min_detection_confidence=DETECTION_CONFIDENCE)
    with vision.HandLandmarker.create_from_options(hand_options) as detector, vision.FaceDetector.create_from_options(face_options) as faces:
        yield detector, faces


def observe_people(frame: np.ndarray, time: float, detector, faces, previous: list,
                   previous_time: float | None, run_hands: bool = True) -> dict:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hands = summarize_hands(detector.detect_for_video(image, round(time * 1000))) if run_hands else None
    detected_faces = faces.detect_for_video(image, round(time * 1000)).detections
    height, width = frame.shape[:2]
    result = {"people_status": "ok" if run_hands else "skipped_dark", "face_status": "ok",
              "face_confidence": max((category.score for face in detected_faces for category in face.categories), default=0.0),
              "face_boxes": [[face.bounding_box.origin_x / width, face.bounding_box.origin_y / height,
                              face.bounding_box.width / width, face.bounding_box.height / height] for face in detected_faces]}
    if hands is not None:
        result.update(hand_landmarks=hands, hand_count=len(hands), hands_inside=float(hands_inside(hands)))
        if hands and previous and previous_time is not None and time > previous_time:
            distances = [min(np.linalg.norm(np.array(hand["points"][0][:2]) - old["points"][0][:2])
                             for old in previous) for hand in hands]
            result["hand_motion"] = float(np.mean(distances) / (time - previous_time))
    return result


def detect_people(rec: Recording, table: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    table = people_columns(table)
    table["face_status"] = "not_sampled"
    targets = np.arange(0, float(table.sample_end.iloc[-1]), 1 / PEOPLE_HZ)
    indices = np.unique(np.searchsorted(table.t.to_numpy(), targets - 1e-6))
    subset = table.iloc[indices[indices < len(table)]]
    previous, previous_time = [], None
    with people_models(model_dir) as (detector, faces):
        for row, frame in iter_sampled_frames(rec, subset):
            time = float(table.at[row, "t"])
            observations = observe_people(frame, time, detector, faces, previous, previous_time)
            for column, value in observations.items():
                table.at[row, column] = value
            previous, previous_time = observations["hand_landmarks"], time
    return table
