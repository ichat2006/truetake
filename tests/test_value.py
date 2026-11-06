import numpy as np
import pandas as pd

from truetake.value import analyze_batch, dtw, edit_distance, farthest_point_order


def table(vector, fingerprint):
    result = pd.DataFrame({"t": np.arange(4) / 2, "sample_end": np.arange(4) / 2 + 0.5,
                           "phash": fingerprint, "mirror_phash": fingerprint})
    result["clip_embedding"] = pd.Series([vector] * 4, dtype=object)
    return result


def test_duplicate_candidate_and_quality_gate():
    tables = {"a": table([1., 0.], "0011223344556677"),
              "copy": table([1., 0.], "0011223344556677"),
              "different": table([0., 1.], "ffffffffffffffff")}
    cards = {"a": {"status": "ACCEPT"}, "copy": {"status": "REVIEW"}, "different": {"status": "ACCEPT"}}
    result = analyze_batch(tables, cards)
    assert len(result["duplicate_candidates"]) == 1
    assert result["duplicate_candidates"][0]["first"] == "a"
    assert {item["clip_id"] for item in result["value_ranking"]} == {"a", "different"}
    assert len(result["map"]) == 3
    assert np.isfinite(np.array(result["distance_matrix"])).all()


def test_farthest_order_starts_typical_then_picks_outlier():
    result = farthest_point_order(np.array([[0., 1., 10.], [1., 0., 9.], [10., 9., 0.]]))
    assert result[0] == (1, 0.)
    assert result[1] == (2, 9.)


def test_distances_and_empty_batches():
    path = np.arange(12).reshape(6, 2)
    assert dtw(path, path) == 0
    assert edit_distance(["smooth", "roll"], ["smooth", "fold-thirds"]) == 0.5
    assert analyze_batch({}, {})["map"] == []
