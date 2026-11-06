"""Run CPU measurements and model tiers on one sequential video decode."""

from pathlib import Path
import shutil
import tempfile
import time

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from truetake.detect_basic import iter_sampled_frames, GRAYSCALE_MAX
from truetake.detect_people import PEOPLE_HZ, people_columns, people_models, observe_people
from truetake.detect_visual import BATCH_SIZE, REVIEW_FRAMES, write_visual_batch
from truetake.evidence import select_evidence, JPEG_QUALITY
from truetake.models import Recording


CLIP_HZ = 1.0
HASH_HZ = 2.0
BLACK_MEAN = 5.0
BLACK_MAX = 10
SCHEDULE_TOLERANCE_S = 1e-6


def scheduled_rows(table: pd.DataFrame, hz: float) -> set[int]:
    targets = np.arange(0, float(table.sample_end.iloc[-1]), 1 / hz)
    positions = np.unique(np.searchsorted(table.t.to_numpy(), targets - SCHEDULE_TOLERANCE_S))
    return {int(table.index[position]) for position in positions if position < len(table)}


def detect_onepass(rec: Recording, table: pd.DataFrame, visual_model, output_dir: str,
                   model_dir: str) -> pd.DataFrame:
    destination = Path(output_dir).resolve()
    protected = (Path(__file__).resolve().parents[2] / "data").resolve()
    if destination == protected or protected in destination.parents:
        raise ValueError("Detector artifacts must be written outside data/.")
    destination.mkdir(parents=True, exist_ok=True)
    thumbnails = destination / "thumbnails"
    thumbnails.mkdir(exist_ok=True)
    started = time.perf_counter()
    table = people_columns(table)
    table["face_status"] = "not_sampled"
    table["clip_status"] = "not_sampled"
    for column in ("brightness", "blur_var", "frame_change", "clothing_similarity", "no_clothing_similarity",
                   "folded_similarity", "crumpled_similarity"):
        table[column] = np.nan
    for column in ("clip_embedding", "phash", "mirror_phash", "thumbnail_path", "detector_stats"):
        table[column] = pd.Series([None] * len(table), index=table.index, dtype=object)
    table["hand_gate_reason"] = "not_scheduled"
    table["clip_gate_reason"] = "not_scheduled"
    people_rows, clip_rows, hash_rows = (scheduled_rows(table, hz) for hz in (PEOPLE_HZ, CLIP_HZ, HASH_HZ))
    preview_rows = set(table.index[np.linspace(0, len(table) - 1, min(REVIEW_FRAMES, len(table))).round().astype(int)])
    _, preprocess, _ = visual_model
    pending, previous_hands, previous_gray, previous_time = [], [], None, None
    counts = {"sampled_frames": 0, "hand_frames": 0, "face_frames": 0, "clip_frames": 0,
              "hash_frames": 0, "hands_skipped_dark": 0, "clip_skipped_dark": 0}
    # Temporary JPEGs keep memory bounded while later measurements determine
    # which eight original-timeline frames Claude needs. Only selected images remain.
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination) as spool:
        with people_models(model_dir) as (hands, faces):
            for row, frame in iter_sampled_frames(rec, table):
                counts["sampled_frames"] += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                table.at[row, "brightness"] = brightness
                table.at[row, "blur_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if previous_gray is not None:
                    table.at[row, "frame_change"] = float(cv2.absdiff(gray, previous_gray).mean() / GRAYSCALE_MAX)
                previous_gray = gray
                black = brightness < BLACK_MEAN and int(gray.max()) <= BLACK_MAX
                if row in people_rows:
                    timestamp = float(table.at[row, "t"])
                    observations = observe_people(frame, timestamp, hands, faces, previous_hands, previous_time, run_hands=not black)
                    for column, value in observations.items():
                        table.at[row, column] = value
                    counts["face_frames"] += 1
                    counts["hands_skipped_dark" if black else "hand_frames"] += 1
                    table.at[row, "hand_gate_reason"] = "nearly_black; unknown, not zero hands" if black else "observable frame"
                    previous_hands, previous_time = observations.get("hand_landmarks") or [], timestamp
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if row in hash_rows:
                    table.at[row, "phash"] = str(imagehash.phash(image))
                    table.at[row, "mirror_phash"] = str(imagehash.phash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
                    counts["hash_frames"] += 1
                if row in preview_rows:
                    path = thumbnails / f"frame_{row:06d}.jpg"
                    image.save(path, quality=JPEG_QUALITY)
                    table.at[row, "thumbnail_path"] = str(path)
                if row in clip_rows:
                    table.at[row, "clip_gate_reason"] = "nearly_black; unknown embedding" if black else "observable frame; hands not required"
                    if black:
                        table.at[row, "clip_status"] = "skipped_dark"
                        counts["clip_skipped_dark"] += 1
                    else:
                        pending.append((row, preprocess(image)))
                        counts["clip_frames"] += 1
                        if len(pending) == BATCH_SIZE:
                            write_visual_batch(table, pending, visual_model)
                if not cv2.imwrite(str(Path(spool) / f"{row}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                    raise ValueError("Could not buffer evidence image.")
        write_visual_batch(table, pending, visual_model)
        table = select_evidence(table)
        selected_dir = destination / "selected"
        selected_dir.mkdir(exist_ok=True)
        for row in table.index[table.claude_selected]:
            path = selected_dir / f"frame_{row:06d}.jpg"
            shutil.copyfile(Path(spool) / f"{row}.jpg", path)
            table.at[row, "thumbnail_path"] = str(path)
    table.at[table.index[0], "detector_stats"] = {
        "mode": "onepass", "analysis_decode_passes": 1, "evidence_decode_passes": 0,
        "source_frames_grabbed": len(rec.frame_times), **counts,
        "scheduled_hand_frames": len(people_rows), "scheduled_clip_frames": len(clip_rows),
        "multipass_clip_frames_at_2hz": len(scheduled_rows(table, HASH_HZ)),
        "clip_hz": CLIP_HZ, "face_hz": PEOPLE_HZ,
        "detector_elapsed_s": round(time.perf_counter() - started, 3),
        "timing_scope": "Decode, hand/face setup, inference and evidence I/O; excludes CLIP model loading, ffprobe, API, follow-up and Rerun export",
        "privacy": "Faces run at 5 Hz even on frames skipped by hand/CLIP gates"}
    return table
