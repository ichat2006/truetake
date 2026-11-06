"""Version-matched local Rerun assets and cache-only per-clip timelines."""
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pandas as pd

RERUN_VERSION = version('rerun-sdk')
ASSET_TIMEOUT_S = 30
LOCK = threading.Lock()
ASSET_URL = None


def viewer_asset(asset: str, output: Path) -> Path:
    """Assets come from the installed wheel's own native HTTP server, never a CDN."""
    global ASSET_URL
    if '..' in Path(asset).parts or Path(asset).is_absolute():raise ValueError('Invalid viewer asset')
    target=output/RERUN_VERSION/(asset or 'index.html')
    with LOCK:
        if target.is_file():return target
        if ASSET_URL is None:
            import rerun as rr
            rr.start_web_viewer_server(port=0)
            ASSET_URL=rr.bindings.get_app_url().rstrip('/')+'/'
            if not ASSET_URL.startswith(('http://127.0.0.1:','http://localhost:')):
                raise RuntimeError('Installed Rerun did not start its local asset server')
        with urlopen(ASSET_URL+(asset or ''),timeout=ASSET_TIMEOUT_S) as response:
            data=response.read()
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    return target


def ensure_recording(run: dict, item: dict) -> Path:
    from truetake.ingest import load_recording
    from truetake.licenses import apply_license
    from truetake.viz_rerun import save_recording
    folder=Path(run['folder'])/'analysis'/item['clip_id']
    path=folder/'card.json'
    if not path.is_file():raise ValueError('Run the review before examining this clip')
    card=apply_license(json.loads(path.read_text()))
    if not card['media_display_allowed']:raise PermissionError('Licence does not allow displaying this media')
    target=Path(run['folder'])/'report'/item['clip_id']/'timeline.rrd'
    with LOCK:
        if target.is_file():return target
        target.parent.mkdir(parents=True,exist_ok=True)
        table=pd.DataFrame(json.loads(Path(card['table_path']).read_text()))
        card['gyro_alignment']=table.iloc[0].get('gyro_alignment')
        rec=replace(load_recording(card['video_path'],card.get('sensors_path')),clip_id=item['clip_id'])
        save_recording(rec,table,card,str(target))
    return target
