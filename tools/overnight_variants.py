"""Add the requested insertion fixtures once, without replacing any data file."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile

from run_manifest_demo import ROOT, read_entries, run

INSERT_AT_S = 8.0
INSERTIONS = {"black_gap": 5.0, "freeze": 6.0}
ENCODER_THREADS = 2
ENCODE_TIMEOUT_S = 600

def insertion_filter(kind: str, seconds: float) -> str:
    # Padding the prefix preserves the exact source tail after the insertion.
    color = ":color=black" if kind == "black_gap" else ""
    mode = "add" if kind == "black_gap" else "clone"
    return (f"[0:v]split=2[a][b];[a]trim=end={INSERT_AT_S},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode={mode}:stop_duration={seconds}{color}[prefix];"
            f"[b]trim=start={INSERT_AT_S},setpts=PTS-STARTPTS[tail];"
            "[prefix][tail]concat=n=2:v=1:a=0[v]")

def create():
    rows = {row['clip_id']:row for row in read_entries()}
    destination = ROOT/'data/clips/variants'
    manifest = destination/'overnight_manifest.csv'
    if manifest.exists():
        print('Existing overnight manifest retained', flush=True)
        return
    parent = rows['basic_fold_1']
    additions=[]
    for kind, seconds in INSERTIONS.items():
        clip_id='basic_fold_1_'+kind
        video=destination/(clip_id+'.mp4')
        if video.exists():
            raise FileExistsError(f'Refusing to replace existing data: {video}')
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='overnight-encode-') as temporary:
            stage=Path(temporary)/video.name
            subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-threads',str(ENCODER_THREADS),
                '-i',str(ROOT/parent['video_path']),'-filter_complex_threads','1',
                '-filter_complex',insertion_filter(kind,seconds),'-map','[v]','-an',
                '-c:v','libx264','-threads',str(ENCODER_THREADS),'-preset','veryfast',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(stage)],check=True,timeout=ENCODE_TIMEOUT_S)
            os.link(stage,video)
        additions.append({**parent,'clip_id':clip_id,'source':'variant',
            'video_path':str(video.relative_to(ROOT)),'sensors_path':'','attack':kind,
            'label_status':'verified','parent_clip_id':'basic_fold_1',
            'notes':f'Verified insertion only: {seconds}s at {INSERT_AT_S}s; pipeline outcome unverified.'})
        print('CREATED',clip_id,flush=True)
    for name in ['P01_102_000_060','P06_102_575_635']:
        additions.append({**rows[name],'clip_id':name+'_wrong_task','source':'variant',
            'task_expected':'folding','attack':'wrong_task','parent_clip_id':name,
            'label_status':'unverified','notes':'Same source, evaluated against folding contract; not independent footage.'})
    fields=list(additions[0])
    with manifest.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(additions)

if __name__=='__main__':
    create()
    print(json.dumps(run(),indent=2))
