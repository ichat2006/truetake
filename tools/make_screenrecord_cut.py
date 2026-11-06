"""Build one licensed EPIC cut fixture from cached content; never call Claude."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pandas as pd
from app.service import catalog,evaluate
from truetake.cached_insert import transfer_insert_cache
from truetake.contracts import load_contract
from truetake.detect_basic import detect_basic
from truetake.export import export_clip
from truetake.frame_table import build_table
from truetake.ingest import load_recording
from truetake.licenses import apply_license
from truetake.report import build_card,write_json
from tools.make_variants_from_own import insertion_filter

ROOT=Path(__file__).resolve().parents[1]
PARENT='P22_109_1851_1911'


def main():
    parent=catalog()[PARENT]
    if parent.get('verdict')!='APPROVED':raise ValueError('Expected an approved parent')
    name=PARENT+'_black_gap'
    folder=ROOT/'outputs/screenrecord'/name;folder.mkdir(parents=True,exist_ok=True)
    video=folder/(name+'.mp4')
    if not video.exists():
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-threads','2','-i',parent['video_path'],
                        '-filter_complex_threads','1','-filter_complex',insertion_filter('black_gap',5.),'-map','[v]',
                        '-an','-c:v','libx264','-threads','2','-preset','veryfast','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(video)],check=True)
    contract_path=ROOT/'contracts/wash_dishes.yaml';contract=load_contract(str(contract_path))
    rec=replace(load_recording(str(video)),clip_id=name)
    original=pd.DataFrame(json.loads(Path(parent['table_path']).read_text()))
    table=transfer_insert_cache(original,build_table(rec,contract))
    # Only inexpensive image statistics are recomputed on the actual encoded frames.
    table=detect_basic(rec,table)
    checks,quality=evaluate(table,contract)
    row={**parent['manifest_entry'],'clip_id':name,'source':'epic','license':'CC-BY-NC-4.0',
         'video_path':str(video.relative_to(ROOT)),'sensors_path':'','parent_clip_id':PARENT,
         'attack':'black_gap','media_modified':'true','notes':'modified for demo',
         'goal_met_expected':'unknown','label_status':'unverified'}
    overlay=ROOT/'outputs/manifest_v2_additions.csv'
    rows=list(csv.DictReader(overlay.open())) if overlay.exists() else []
    if not any(r['clip_id']==name for r in rows):
        rows.append(row)
        with overlay.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    card=apply_license({**build_card(name,table,checks,quality),'video_path':str(video),'sensors_path':None,
        'source':'epic','license':'CC-BY-NC-4.0','media_modified':True,'manifest_entry':row,
        'task_contract':str(contract_path),'demo_gap':[8.,13.],
        'cache_provenance':{'parent_clip_id':PARENT,'reused':'unchanged frame content and retimestamped goal evidence',
                            'new_measurements':'encoded brightness, sharpness, frame change','api_calls':0},
        'card_path':str(folder/'card.json'),'table_path':str(folder/'frames.json'),
        'rerun_path':str(folder/'timeline.rrd')},contract)
    card['export']=export_clip(rec,card,str(folder/'exports'),table)
    table.to_json(card['table_path'],orient='records',double_precision=15)
    write_json(card['card_path'],card)
    write_json(ROOT/'outputs/manifest_run'/(name+'.json'),{'clip_id':name,'manifest_entry':row,'card':card,'errors':[],'status':'complete'})
    print(json.dumps({'clip_id':name,'verdict':card['verdict'],'reason':card['decision_reasons'],
                      'gap':[8,13],'applied_cuts':card['export'].get('plan',{}).get('applied_cuts'),
                      'export':card['export'].get('video_path'),'api_calls':0}),flush=True)


if __name__=='__main__':main()
