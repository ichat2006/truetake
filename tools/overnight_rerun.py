"""Regenerate the requested cached timelines, without detector or AI calls."""
import json
from pathlib import Path
import pandas as pd
from truetake.ingest import load_recording
from truetake.viz_rerun import save_recording

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    regenerated=[]
    for name in ['basic_fold_1_black_gap','splice_P03_109_P04_104','P01_102_000_060']:
        card=json.loads((ROOT/'outputs/manifest_run'/(name+'.json')).read_text())['card']
        rec=load_recording(card['video_path'],card.get('sensors_path'))
        from dataclasses import replace
        rec=replace(rec,clip_id=name)
        table=pd.DataFrame(json.loads(Path(card['table_path']).read_text()))
        save_recording(rec,table,card,card['rerun_path'])
        regenerated.append({'clip_id':name,'verdict':card.get('verdict'),
                            'path':card['rerun_path'],'applied_cuts':(card.get('export') or {}).get('plan',{}).get('applied_cuts',[])})
        print('REGENERATED',name,flush=True)
    (ROOT/'outputs/overnight/step7_rerun.json').write_text(json.dumps(regenerated,indent=2))
