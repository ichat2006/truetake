"""Create insertion fixtures from explicitly self-recorded footage only."""
import argparse
import csv
from pathlib import Path
import subprocess
from overnight_variants import insertion_filter

ROOT=Path(__file__).resolve().parents[1]

def create(video,name,split):
    from truetake.licenses import permissions
    if not permissions('own')['modify']:raise ValueError('Own policy disallows variants')
    manifest=ROOT/'data/manifest_v2.csv'
    with manifest.open(newline='') as handle:
        reader=csv.DictReader(handle);fields=list(reader.fieldnames);rows=list(reader)
    parent=next((row for row in rows if row['clip_id']==name),None)
    if parent and parent['source']!='own':raise ValueError('Catalog clip is not self-recorded')
    if Path(name).name!=name or not name:raise ValueError('Use a simple clip name')
    video=Path(video).resolve()
    if not video.is_file():raise FileNotFoundError(video)
    if not parent:
        parent={key:'' for key in fields}
        parent.update(clip_id=name,source='own',collection='own',participant='self',video_path=str(video),
            license='own',split=split,task_expected='unspecified',goal_met_expected='unknown',
            attack='none',label_status='unverified',media_modified='false')
        rows.append(parent)
    additions=[];out=ROOT/'outputs/own_variants';out.mkdir(parents=True,exist_ok=True)
    for kind,seconds in [('black_gap',5.),('freeze',6.)]:
        clip_id=name+'_'+kind;target=out/(clip_id+'.mp4')
        if target.exists() or any(row['clip_id']==clip_id for row in rows):raise FileExistsError(clip_id)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-threads','2','-i',str(video),
            '-filter_complex_threads','1','-filter_complex',insertion_filter(kind,seconds),'-map','[v]',
            '-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
        additions.append({**parent,'clip_id':clip_id,'video_path':str(target),'source':'own','collection':'own',
            'license':'own','sensors_path':'','attack':kind,'parent_clip_id':name,'media_modified':'true',
            'label_status':'verified','notes':'Verified insertion only; no fabricated retimed sensors.'})
    with manifest.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows+additions)
    return additions

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Use only footage you filmed; this declares that provenance.')
    parser.add_argument('video');parser.add_argument('--name',required=True);parser.add_argument('--split',choices=['train','test'],default='train')
    args=parser.parse_args()
    for row in create(args.video,args.name,args.split):print(row['clip_id'],row['video_path'])
