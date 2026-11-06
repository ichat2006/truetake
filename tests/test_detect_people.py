from types import SimpleNamespace

from truetake.detect_people import hands_inside, summarize_hands


def test_all_landmarks_and_two_hands_required():
    hand = {"points": [[0.5, 0.5, 0.0]] * 21}
    assert hands_inside([hand, hand])
    assert not hands_inside([hand])
    clipped = {"points": [[0.5, 0.5, 0.0]] * 20 + [[0.01, 0.5, 0.0]]}
    assert not hands_inside([hand, clipped])
    assert not hands_inside([hand, {"points": hand["points"][:20]}])


def test_empty_detections_are_real_zero_hands():
    result = SimpleNamespace(hand_landmarks=[], handedness=[])
    assert summarize_hands(result) == []
    assert not hands_inside([])
