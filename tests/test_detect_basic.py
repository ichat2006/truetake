import cv2
import numpy as np
import pandas as pd
import pytest

from truetake.detect_basic import detect_basic, iter_sampled_frames
from truetake.models import Recording


def test_brightness_blur_and_change(monkeypatch):
    black = np.zeros((32, 32, 3), np.uint8)
    white = np.full_like(black, 255)
    checker = np.repeat(((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    blurred = cv2.GaussianBlur(checker, (5, 5), 0)
    frames = [black, black, white, checker, blurred]
    monkeypatch.setattr("truetake.detect_basic.iter_sampled_frames", lambda *args: enumerate(frames))
    original = pd.DataFrame({"t": np.arange(5) / 10})
    table = detect_basic(None, original)
    assert table.brightness.iloc[0] == 0
    assert table.brightness.iloc[2] == 255
    assert np.isnan(table.frame_change.iloc[0])
    assert table.frame_change.iloc[1] == 0
    assert table.frame_change.iloc[2] == 1
    assert table.blur_var.iloc[3] > table.blur_var.iloc[4]
    assert "brightness" not in original


def test_rotation_is_applied_once_and_decoder_is_released(monkeypatch):
    frame = np.zeros((2, 3, 3), np.uint8)
    frame[0, 0] = 255

    class Capture:
        released = False
        def isOpened(self): return True
        def set(self, key, value):
            assert key == cv2.CAP_PROP_ORIENTATION_AUTO and value == 0
        def grab(self): return True
        def retrieve(self): return True, frame.copy()
        def release(self): self.released = True

    capture = Capture()
    monkeypatch.setattr("truetake.detect_basic.cv2.VideoCapture", lambda path: capture)
    rec = Recording("rotated", "rotated.mp4", np.array([0.]), {"rotation": 90}, None)
    decoded = list(iter_sampled_frames(rec, pd.DataFrame({"source_frame": [0]})))[0][1]
    assert decoded.shape == (3, 2, 3)
    assert (decoded[2, 0] == 255).all()
    assert capture.released


def test_truncated_decode_is_an_error_and_releases_capture(monkeypatch):
    class Capture:
        released = False
        def isOpened(self): return True
        def set(self, *args): pass
        def grab(self): return False
        def release(self): self.released = True

    capture = Capture()
    monkeypatch.setattr("truetake.detect_basic.cv2.VideoCapture", lambda path: capture)
    rec = Recording("bad", "bad.mp4", np.array([0.]), {"rotation": 0}, None)
    with pytest.raises(ValueError, match="decoding stopped"):
        list(iter_sampled_frames(rec, pd.DataFrame({"source_frame": [0]})))
    assert capture.released


@pytest.mark.parametrize("rotation,expected", [(0, (540, 960, 3)), (90, (1707, 960, 3))])
def test_display_resolution_keeps_analysis_default_and_rotation(monkeypatch, rotation, expected):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    class Capture:
        def isOpened(self): return True
        def set(self, *args): pass
        def grab(self): return True
        def retrieve(self): return True, frame.copy()
        def release(self): pass
    monkeypatch.setattr("truetake.detect_basic.cv2.VideoCapture", lambda path: Capture())
    rec = Recording("sample", "sample.mp4", np.array([0.]), {"rotation": rotation}, None)
    table = pd.DataFrame({"source_frame": [0]})
    display = next(iter_sampled_frames(rec, table, max_width=960))[1]
    analysis = next(iter_sampled_frames(rec, table))[1]
    assert display.shape == expected
    assert analysis.shape[1] == 320
