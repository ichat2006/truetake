"""Scene, hand-path, and strategy distances; approved-only value selection."""

import numpy as np
import pandas as pd


PATH_STEPS = 64
MIN_HAND_SAMPLES = 4
MAX_HAND_GAP_S = 1.0
REDUNDANCY_DISTANCE = 0.35
DUPLICATE_HASH_BITS = 4
HASH_FRAMES = 8
MIN_DUPLICATE_FRAMES = 4
EPSILON = 1e-8


def suggest_next(cards: dict[str, dict], wanted: list[dict]) -> str:
    """Describe the least-covered requested variation using approved review tags."""
    accepted = [card for card in cards.values() if card["status"] == "ACCEPT" and not card.get("fixture_kind")]
    if not wanted:
        return "Record a different setup or technique to expand the approved collection."
    coverage = []
    for variation in wanted:
        count = sum(all((card.get("claude_review") or {}).get("tags", {}).get(key) == value
                        for key, value in variation.items()) for card in accepted)
        coverage.append((count, variation))
    count, variation = min(coverage, key=lambda item: item[0])
    styles = {"fold-half-short": "folded in half", "fold-thirds": "folded in thirds", "roll": "rolled"}
    garment = "T-shirt" if variation.get("garment") == "tshirt" else variation.get("garment", "garment")
    style = styles.get(variation.get("style"), str(variation.get("style", "with a different technique")).replace("-", " "))
    surface = variation.get("surface", "different surface")
    return f"Record a {garment} {style} on a {surface} — {count} approved examples match this variation."


def scene_vector(table: pd.DataFrame) -> np.ndarray | None:
    vectors = table.clip_embedding.dropna().tolist() if "clip_embedding" in table else []
    if not vectors:
        return None
    vector = np.mean(vectors, axis=0)
    return vector / max(np.linalg.norm(vector), EPSILON)


def hand_path(table: pd.DataFrame) -> np.ndarray | None:
    if "hand_landmarks" not in table:
        return None
    times, values = [], []
    for _, row in table.iterrows():
        hands = row.hand_landmarks
        if not isinstance(hands, list):
            continue
        labels = {hand["label"]: hand for hand in hands}
        if not {"Left", "Right"} <= labels.keys():
            continue
        times.append(float(row.t))
        values.append([coordinate for side in ("Left", "Right") for point in (0, 4, 8)
                       for coordinate in labels[side]["points"][point][:2]])
    if len(times) < MIN_HAND_SAMPLES or np.diff(times).max() > MAX_HAND_GAP_S:
        return None
    # Reject sparse fragments rather than invent a full task trajectory.
    if times[-1] - times[0] < float(table.sample_end.iloc[-1]) / 2:
        return None
    target = np.linspace(times[0], times[-1], PATH_STEPS)
    array = np.array(values)
    return np.column_stack([np.interp(target, times, array[:, index]) for index in range(array.shape[1])])


def dtw(first: np.ndarray, second: np.ndarray) -> float:
    previous = np.full(len(second) + 1, np.inf)
    previous[0] = 0.
    for a in first:
        current = np.full(len(second) + 1, np.inf)
        for index, b in enumerate(second, start=1):
            current[index] = np.linalg.norm(a - b) + min(previous[index], current[index - 1], previous[index - 1])
        previous = current
    return float(previous[-1] / (len(first) + len(second)))


def edit_distance(first: list[str], second: list[str]) -> float:
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(first), len(second), 1)


def farthest_point_order(distance: np.ndarray) -> list[tuple[int, float]]:
    if not len(distance):
        return []
    selected = [int(np.argmin(distance.mean(axis=1)))]
    result = [(selected[0], 0.0)]  # Representative seed, not the most novel clip.
    while len(selected) < len(distance):
        nearest = distance[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        index = int(np.argmax(nearest))
        result.append((index, float(nearest[index])))
        selected.append(index)
    return result


def _hash_distance(first: pd.DataFrame, second: pd.DataFrame) -> float | None:
    if "phash" not in first or "phash" not in second:
        return None
    a, b = first[first.phash.notna()], second[second.phash.notna()]
    if min(len(a), len(b)) < MIN_DUPLICATE_FRAMES:
        return None
    length = min(HASH_FRAMES, len(a), len(b))
    a = a.iloc[np.linspace(0, len(a) - 1, length).round().astype(int)]
    b = b.iloc[np.linspace(0, len(b) - 1, length).round().astype(int)]
    distances = []
    for column in ("phash", "mirror_phash"):
        if column in b and b[column].notna().all():
            distances.append(np.median([(int(x, 16) ^ int(y, 16)).bit_count() for x, y in zip(a.phash, b[column])]))
    return float(min(distances)) if distances else None


def analyze_batch(tables: dict[str, pd.DataFrame], cards: dict[str, dict]) -> dict:
    ids = [clip for clip, table in tables.items() if scene_vector(table) is not None]
    scenes = {clip: scene_vector(tables[clip]) for clip in ids}
    paths = {clip: hand_path(tables[clip]) for clip in ids}
    strategies = {}
    for clip in ids:
        review = tables[clip].iloc[0].get("claude_review") or {}
        strategies[clip] = [step["name"] for step in review.get("steps", [])] or None
    size = len(ids)
    views = {name: np.full((size, size), np.nan) for name in ("scene", "hands", "strategy")}
    duplicate_pairs = []
    for i, first in enumerate(ids):
        for j in range(i + 1, size):
            second = ids[j]
            views["scene"][i, j] = views["scene"][j, i] = max(0., 1 - float(scenes[first] @ scenes[second]))
            if paths[first] is not None and paths[second] is not None:
                views["hands"][i, j] = views["hands"][j, i] = dtw(paths[first], paths[second])
            if strategies[first] and strategies[second]:
                views["strategy"][i, j] = views["strategy"][j, i] = edit_distance(strategies[first], strategies[second])
            hash_distance = _hash_distance(tables[first], tables[second])
            if hash_distance is not None and hash_distance <= DUPLICATE_HASH_BITS:
                duplicate_pairs.append({"first": first, "second": second, "median_hash_bits": hash_distance,
                                        "kind": "visual duplicate candidate; human confirmation required"})
    normalized = []
    scales = {}
    for name, matrix in views.items():
        finite = matrix[np.isfinite(matrix) & (matrix > EPSILON)]
        scales[name] = float(np.median(finite)) if len(finite) else 1.0
        normalized.append(matrix / scales[name])
    distance = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            available = [matrix[i, j] for matrix in normalized if np.isfinite(matrix[i, j])]
            distance[i, j] = distance[j, i] = float(np.mean(available))
    # Classical MDS: deterministic 2D layout for the small demo batch.
    center = np.eye(size) - np.ones((size, size)) / max(size, 1)
    gram = -0.5 * center @ (distance ** 2) @ center
    coordinates = np.zeros((size, 2))
    if size:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        for dimension, index in enumerate(np.argsort(eigenvalues)[::-1][:2]):
            coordinates[:, dimension] = eigenvectors[:, index] * np.sqrt(max(0., eigenvalues[index]))
    accepted = [i for i, clip in enumerate(ids) if cards[clip]["status"] == "ACCEPT"]
    ranking = []
    for rank, (index, novelty) in enumerate(farthest_point_order(distance[np.ix_(accepted, accepted)]), start=1):
        clip = ids[accepted[index]]
        ranking.append({"clip_id": clip, "rank": rank, "novelty": novelty,
                        "reason": "representative seed" if rank == 1 else "farthest from previously selected approved clips"})
    redundant = []
    for i in accepted:
        neighbors = [j for j in accepted if j != i]
        if neighbors:
            j = min(neighbors, key=lambda j: distance[i, j])
            if distance[i, j] < REDUNDANCY_DISTANCE:
                redundant.append({"clip_id": ids[i], "nearest_clip": ids[j], "distance": float(distance[i, j])})
    scene_distances = views["scene"].copy()
    np.fill_diagonal(scene_distances, 0)
    scene_means = scene_distances.mean(axis=1) if size else np.array([])
    maximum = max(float(scene_means.max()), EPSILON) if size else 1.
    return {"clip_ids": ids, "distance_matrix": distance.tolist(), "view_scales": scales,
            "scene_distinctness": {clip: round(100 * float(scene_means[i]) / maximum, 1) for i, clip in enumerate(ids)},
            "available_views": {clip: [name for name, present in (("scene", True), ("hands", paths[clip] is not None),
                                                                          ("strategy", strategies[clip] is not None)) if present] for clip in ids},
            "map": [{"clip_id": clip, "x": float(coordinates[i, 0]), "y": float(coordinates[i, 1]),
                     "status": cards[clip]["status"]} for i, clip in enumerate(ids)],
            "value_ranking": ranking, "redundant": redundant, "duplicate_candidates": duplicate_pairs,
            "note": "Map shows all clips for inspection; training-value ranking and redundancy include accepted clips only. Thresholds are provisional."}
