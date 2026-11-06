"""Recompute contract decisions and exports using saved tables and reviews only."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from truetake.contracts import load_contract
from truetake.export import export_clip, omitted_entry
from truetake.ingest import load_recording
from truetake.models import CheckResult
from truetake.presentation import badge, plain_reason
from truetake.quality import check_quality
from truetake.verify_gyro import check_verification
from truetake.verdicts import apply_verdict
from truetake.viz_rerun import save_recording

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'outputs/manifest_run'
OUT=ROOT/'outputs/hand_presence'

def cache_snapshot() -> dict:
    paths=set()
    for root in [ROOT/'outputs/demo_work/runs',ROOT/'outputs/demo']:
        for pattern in ['**/detectors*.json','**/detectors*.sha256','**/claude_cache/*.json']:
            paths.update(root.glob(pattern))
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}

def recompute(table: pd.DataFrame, card: dict, contract: dict) -> tuple[pd.DataFrame,dict]:
    table=table.copy()
    review=table.iloc[0].get('claude_review')
    goal=(review or {}).get('goal',{}).get('verdict')
    if goal not in ('yes','no','unsure',None):
        raise ValueError('Unsupported cached goal.verdict; inspect its schema before mapping')
    if goal!=(card.get('claude_review') or {}).get('goal',{}).get('verdict'):
        raise ValueError('Card and table cached goals disagree')
    original=json.dumps(review,sort_keys=True)
    table.at[0,'contract']=contract
    checks,quality=check_quality(table)
    verification=(check_verification(table) if table.sensor_present.any() else
                  [CheckResult('verification.gyro',None,0.,'no motion data',[])])
    checks+=verification
    checks,quality=apply_verdict(table,checks,quality)
    result={**card,**quality,'checks':[asdict(check) for check in checks],
            'verification':asdict(verification[0]),
            'decision_recompute':{'detector_calls':0,'api_calls':0,
                'claude_source':'cached review' if review else 'no cached review available',
                'cached_goal_verdict':goal,'new_contract_applied_to_measurements_only':True}}
    if json.dumps(table.iloc[0].get('claude_review'),sort_keys=True)!=original:
        raise RuntimeError('A cached review changed during decision recomputation')
    return table,result

def atomic_json(path: Path,value: dict):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)

def main(rerun: bool=False):
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=cache_snapshot()
    rows=[]
    for path in sorted(RESULTS.glob('*.json')):
        if path.name=='batch.json':continue
        payload=json.loads(path.read_text())
        card=payload.get('card')
        if not card:continue
        table=pd.DataFrame(json.loads(Path(card['table_path']).read_text()))
        contract=load_contract(card.get('task_contract') or str(ROOT/'contracts/mercor_folding.yaml'))
        table,card=recompute(table,card,contract)
        rec=None
        omitted=omitted_entry(card)
        if omitted:
            card['export']=omitted
        else:
            rec=replace(load_recording(card['video_path'],card.get('sensors_path')),clip_id=card['clip_id'])
            card['export']=export_clip(rec,card,str(ROOT/'outputs/demo_work/exports_cached'),table)
        if rerun:
            rec=rec or replace(load_recording(card['video_path'],card.get('sensors_path')),clip_id=card['clip_id'])
            save_recording(rec,table,card,card['rerun_path'])
        Path(card['table_path']).write_text(table.to_json(orient='records',indent=2,double_precision=15))
        atomic_json(Path(card['card_path']),card)
        payload['card']=card
        atomic_json(path,payload)
        hand=next(r for r in card['measured']['rules'] if r['measurement']=='hand_present')
        row={'clip_id':card['clip_id'],'verdict':badge(card),'reason':plain_reason(card),
             'hand_present':hand['value'],'failure_reasons':card['failure_reasons'],
             'review_reasons':card['review_reasons'],'cached_goal':card['decision_recompute']['cached_goal_verdict'],
             'export_status':card['export']['export_status'],
             'applied_cuts':card['export'].get('plan',{}).get('applied_cuts',[])}
        rows.append(row)
        atomic_json(OUT/(card['clip_id']+'.json'),row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    if cache_snapshot()!=snapshot:
        raise RuntimeError('Detector or Claude cache contents changed')
    summary={'counts':dict(Counter(row['verdict'] for row in rows)),
             'detector_calls':0,'api_calls':0,'unchanged_cache_files':len(snapshot),'rows':rows}
    atomic_json(OUT/'after.json',summary)
    subprocess.run(['uv','run','python','tools/build_demo_report.py'],cwd=ROOT,check=True)
    print('FINAL COUNTS',json.dumps(summary['counts']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun',action='store_true',help='Refresh cached-evidence Rerun views too')
    main(parser.parse_args().rerun)
