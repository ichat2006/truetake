"""CPU CLIP measurements, visual fingerprints, and evidence thumbnails."""

from pathlib import Path

import cv2
import imagehash
import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import torch

from truetake.detect_basic import iter_sampled_frames
from truetake.models import Recording


CLIP_HZ = 2.0
BATCH_SIZE = 8
CPU_THREADS = 2
REVIEW_FRAMES = 12
MODEL_NAME = "ViT-B-32"
TEXTS = ["a garment lying on a work surface", "an empty work surface with no garment",
         "a neatly folded garment in a compact rectangle", "a crumpled unfolded garment"]


def load_visual_model(weights: str):
    if not Path(weights).is_file():
        raise ValueError("CLIP weights are missing; download outputs/models/ViT-B-32.pt first.")
    torch.set_num_threads(CPU_THREADS)
    # The OpenAI loader understands the official TorchScript archive format.
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained="openai", device="cpu", cache_dir=str(Path(weights).resolve().parent))
    model.eval()
    tokens = open_clip.get_tokenizer(MODEL_NAME)(TEXTS)
    with torch.inference_mode():
        text = model.encode_text(tokens)
        text /= text.norm(dim=-1, keepdim=True)
    return model, preprocess, text


def write_visual_batch(table: pd.DataFrame, pending: list, visual_model) -> None:
    """Write one small CPU batch into the shared table, then release it."""
    if not pending:
        return
    model, _, text = visual_model
    with torch.inference_mode():
        embeddings = model.encode_image(torch.stack([item[1] for item in pending]))
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        similarities = embeddings @ text.T
    for (row, _), embedding, scores in zip(pending, embeddings.tolist(), similarities.tolist()):
        table.at[row, "clip_embedding"] = embedding
        table.at[row, "clip_status"] = "ok"
        for column, score in zip(("clothing_similarity", "no_clothing_similarity", "folded_similarity", "crumpled_similarity"), scores):
            table.at[row, column] = score
    pending.clear()


def detect_visual(rec: Recording, table: pd.DataFrame, visual_model, output_dir: str) -> pd.DataFrame:
    table = table.copy()
    table["clip_status"] = "not_sampled"
    for column in ("clothing_similarity", "no_clothing_similarity", "folded_similarity", "crumpled_similarity"):
        table[column] = np.nan
    for column in ("clip_embedding", "phash", "mirror_phash", "thumbnail_path"):
        table[column] = pd.Series([None] * len(table), index=table.index, dtype=object)
    times = table.t.to_numpy()
    targets = np.arange(0, float(table.sample_end.iloc[-1]), 1 / CLIP_HZ)
    selected = np.unique(np.searchsorted(times, targets - 1e-6))
    selected = selected[selected < len(table)]
    evidence = set(np.linspace(0, len(table) - 1, min(REVIEW_FRAMES, len(table))).round().astype(int))
    subset = table.iloc[sorted(set(selected) | evidence)]
    destination = Path(output_dir) / "thumbnails"
    destination.mkdir(parents=True, exist_ok=True)
    _, preprocess, _ = visual_model
    pending = []

    for row, frame in iter_sampled_frames(rec, subset):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if row in evidence:
            path = destination / f"frame_{row:06d}.jpg"
            image.save(path, quality=85)
            table.at[row, "thumbnail_path"] = str(path.resolve())
        if row in selected:
            table.at[row, "phash"] = str(imagehash.phash(image))
            table.at[row, "mirror_phash"] = str(imagehash.phash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
            pending.append((row, preprocess(image)))
            if len(pending) == BATCH_SIZE:
                write_visual_batch(table, pending, visual_model)
    write_visual_batch(table, pending, visual_model)
    return table
