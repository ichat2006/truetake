"""Exercise real preview encoding and privacy blur without using real footage."""

import hashlib
import shutil
import subprocess

import cv2
import pytest

from truetake.preview import browser_preview


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def sharpness_at(path, seconds):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
        assert ok
        return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    finally:
        capture.release()


def test_privacy_preview_blurs_only_requested_times_and_preserves_source(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=2",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(source)], check=True, capture_output=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert browser_preview(str(source), str(tmp_path / "previews")) == str(source)
    target = browser_preview(str(source), str(tmp_path / "previews"), [[0., .9]])
    assert target != str(source)
    assert sharpness_at(target, .5) < sharpness_at(source, .5) / 10
    assert sharpness_at(target, 1.5) > sharpness_at(source, 1.5) / 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert browser_preview(str(source), str(tmp_path / "previews"), [[0., .9]]) == target
