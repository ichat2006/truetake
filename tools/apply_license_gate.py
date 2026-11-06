"""Attach rights to cached results, rebuild public assets, and print exclusions."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from truetake.licenses import apply_license
from truetake.export import omitted_entry
from truetake.contracts import load_contract

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    with (ROOT/'data/manifest_v2.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
    out=ROOT/'outputs/licenses';out.mkdir(parents=True,exist_ok=True);audit=[]
    for row in rows:
        path=ROOT/'outputs/manifest_run'/(row['clip_id']+'.json')
        payload=json.loads(path.read_text()) if path.exists() else None
        card=payload['card'] if payload else {'clip_id':row['clip_id'],'status':'REVIEW'}
        before=(card.get('status'),card.get('verdict'),card.get('decision_reasons'))
        card.update(source=row['source'],license=row['license'],manifest_entry=row,media_modified=row['media_modified'])
        card=apply_license(card,load_contract(str(ROOT/'contracts/mercor_folding.yaml')))
        if not card['license_permissions']['modify']:card['export']=omitted_entry(card)
        assert before==(card.get('status'),card.get('verdict'),card.get('decision_reasons'))
        if payload:
            payload['card']=card;path.write_text(json.dumps(payload,indent=2));Path(card['card_path']).write_text(json.dumps(card,indent=2))
        reason=card['public_exclusion']
        if not (ROOT/row['video_path']).is_file():reason=(reason+'; ' if reason else '')+'Video missing'
        if not card['license_permissions']['modify']:reason=(reason+'; ' if reason else '')+'Modified exports withheld'
        audit.append({'clip_id':row['clip_id'],'source':row['source'],'license':card['license'],
            'eligibility':card['eligibility']['line'],'excluded_from_public':reason or 'None'})
    with (out/'eligibility.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    for item in audit:print(' | '.join(item.values()),flush=True)
    subprocess.run([sys.executable,str(ROOT/'tools/build_demo_report.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'docs/package_submission.py')],cwd=ROOT,check=True)
