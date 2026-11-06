"""Create browser-playable derivatives without changing review source videos."""

import json
from pathlib import Path
import subprocess


PREVIEW_WIDTH = 640
ENCODER_THREADS = 2
CRF = 24
TIMEOUT_S = 90
BLUR_RADIUS = 24


def browser_preview(video_path: str, output_dir: str, blur_intervals: list | None = None) -> str:
    source = Path(video_path).resolve()
    blur_intervals = [[float(a), float(b)] for a, b in (blur_intervals or [])]
    root = Path(output_dir).resolve()
    data = (Path(__file__).resolve().parents[2] / "data").resolve()
    if root == data or data in root.parents:
        raise ValueError("Previews must be written outside data/.")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt", "-of", "json", str(source)],
                           capture_output=True, text=True, check=True, timeout=TIMEOUT_S)
    stream = json.loads(probe.stdout)["streams"][0]
    if not blur_intervals and stream["codec_name"] == "h264" and stream["pix_fmt"] in ("yuv420p", "yuvj420p"):
        return str(source)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{source.stem}.mp4"
    metadata = target.with_suffix(".preview.json")
    stat = source.stat()
    signature = {"source": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "width": PREVIEW_WIDTH, "crf": CRF, "blur_intervals": blur_intervals}
    if target.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == signature:
        return str(target)
    temporary = target.with_name(target.stem + "_encoding.mp4")
    filters = f"scale='min({PREVIEW_WIDTH},iw)':-2"
    if blur_intervals:
        enabled = "+".join(f"between(t,{a:.6f},{b:.6f})" for a, b in blur_intervals)
        filters += f",boxblur=luma_radius={BLUR_RADIUS}:luma_power=3:chroma_radius=12:enable='{enabled}'"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-map", "0:v:0", "-an", "-vf", filters,
                    "-c:v", "libx264", "-threads", str(ENCODER_THREADS), "-preset", "veryfast",
                    "-crf", str(CRF), "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-movflags", "+faststart", str(temporary)],
                   capture_output=True, text=True, check=True, timeout=TIMEOUT_S)
    temporary.replace(target)
    metadata.write_text(json.dumps(signature), encoding="utf-8")
    return str(target)
