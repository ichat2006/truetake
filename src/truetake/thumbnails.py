"""Small, real midpoint images for browser previews and clip maps."""
from pathlib import Path
import subprocess

THUMBNAIL_WIDTH = 320
THUMBNAIL_TIMEOUT_S = 30


def midpoint_thumbnail(video_path: str, output_path: str, duration_s: float) -> str:
    target = Path(output_path)
    if target.is_file() and target.stat().st_size:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.encoding.jpg')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1',
                    '-ss', str(max(0., duration_s / 2)), '-i', str(video_path),
                    '-frames:v', '1', '-vf', f'scale={THUMBNAIL_WIDTH}:-2',
                    '-threads', '1', str(temporary)], check=True, timeout=THUMBNAIL_TIMEOUT_S)
    temporary.replace(target)
    return str(target)
