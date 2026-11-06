"""Apply current YAML decision policy to cached evidence; train-only band sanity check."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import yaml

from truetake.contracts import load_contract
from truetake.measured import defaults
from truetake.quality import check_quality
from truetake.verify_gyro import check_verification
from truetake.verdicts import apply_verdict
from truetake.export import omitted_entry

ROOT=Path(__file__).resolve().parents[1]

def compute(train_only=False,persist=False):
    rows=[]
    for path in (ROOT/'outputs/manifest_run').glob('*.json'):
        if path.name=='batch.json':continue
        payload=json.loads(path.read_text());card=payload.get('card') or {}
        if not card or card.get('fixture_kind'):continue
        if train_only and card.get('split')!='train':continue
        table=pd.DataFrame(json.loads(Path(card['table_path']).read_text()))
        contract=load_contract(card.get('task_contract') or str(ROOT/'contracts/mercor_folding.yaml'))
        table.at[0,'contract']=contract
        checks,quality=check_quality(table)
        if card.get('sensor_present'):
            checks+=check_verification(table)
        else:
            from truetake.models import CheckResult
            checks+=[CheckResult('verification.gyro',None,0.,'no motion data',[])]
        checks,quality=apply_verdict(table,checks,quality)
        rows.append({'clip_id':card['clip_id'],'split':card['split'],'source':card['source'],
                     'verdict':quality['verdict'],'reason':quality['decision_reasons'][0]})
        if persist:
            card.update(quality,checks=[asdict(check) for check in checks])
            if card['status']=='REJECT':card['export']=omitted_entry(card)
            elif card['status']=='REVIEW' and card.get('export',{}).get('video_path'):
                card['export'].update(export_status='retained_pending',eligible_for_training=False)
            table.to_json(card['table_path'],orient='records',indent=2,double_precision=15)
            Path(card['card_path']).write_text(json.dumps(card,indent=2))
            path.write_text(json.dumps(payload,indent=2))
    return rows

if __name__=='__main__':
    path=ROOT/'contracts/defaults.yaml';settings=yaml.safe_load(path.read_text())
    policy=settings['review_sanity']
    audit=[]
    for attempt in range(3):
        rows=compute(train_only=True)
        reviews=[row for row in rows if row['verdict']=='NEEDS REVIEW']
        audit.append({'attempt':attempt,'train_count':len(rows),'review_count':len(reviews),
                      'reasons':dict(Counter(row['reason'].split(':')[0] for row in reviews))})
        if len(reviews)<=len(rows)*policy['max_train_review_fraction']:break
        if attempt==0:
            settings['review_band']*=policy['halve_factor'];settings['review_band_seconds']*=policy['halve_factor']
        elif attempt==1:settings['close_calls_enabled']=False
        else:break
        path.write_text(yaml.safe_dump(settings,sort_keys=False))
    rows=compute(persist=True)
    (ROOT/'outputs/overnight/step3_sanity.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'outputs/overnight/step3_verdicts.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'train_audit':audit,'all_counts':dict(Counter(row['verdict'] for row in rows))},indent=2))
