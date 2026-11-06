"""Audit TEST manifest labels against saved verdicts without running the pipeline."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NEGATIVE_ATTACKS={'splice','swap','sensor_swap','slow2x','wrong_task','dark'}

def expected_verdict(row):
    attack=row.get('attack','').strip().lower()
    if attack=='controlled_reupload':return 'DUPLICATE'
    if attack in NEGATIVE_ATTACKS:return 'REJECTED'
    goal=str(row.get('goal_met_expected','')).strip().lower()
    if goal in {'yes','true','1'}:return 'APPROVED'
    if goal in {'no','false','0'}:return 'REJECTED'
    return 'UNKNOWN'

def compare(rows,cards):
    results=[]
    for row in rows:
        if row.get('split')!='test' or row.get('label_status') not in {'verified','unverified'}:continue
        card=cards.get(row['clip_id'])
        verdict=(card.get('verdict') or {'ACCEPT':'APPROVED','REVIEW':'NEEDS REVIEW','REJECT':'REJECTED'}.get(card.get('status'))) if card else 'MISSING RESULT'
        if card and card.get('fixture_kind'):verdict='DUPLICATE'
        expected=expected_verdict(row)
        agree=None if verdict in {'NEEDS REVIEW','MISSING RESULT'} or expected=='UNKNOWN' else verdict==expected
        results.append({'clip_id':row['clip_id'],'source':row.get('source'),
            'label_status':row['label_status'],'expected':expected,'verdict':verdict,'agree':agree})
    known=[r for r in results if r['agree'] is not None]
    reviews=sum(r['verdict']=='NEEDS REVIEW' for r in results)
    line=f"Agreed with labels on {sum(r['agree'] for r in known)} of {len(known)} comparable test clips; NEEDS REVIEW counted separately ({reviews})."
    return {'line':line,'rows':results,'label_status_counts':dict(Counter(r['label_status'] for r in results)),
            'comparable_label_status_counts':dict(Counter(r['label_status'] for r in known)),
            'unknown_expected':sum(r['expected']=='UNKNOWN' for r in results),
            'missing_results':sum(r['verdict']=='MISSING RESULT' for r in results)}

def markdown(result,public=False):
    from truetake.licenses import catalog_metadata,apply_license
    metadata=catalog_metadata()
    rows=[r for r in result['rows'] if not public or apply_license(metadata.get(r['clip_id'],{}))['public_media_allowed']]
    text=['# Test-label agreement','',result['line'],'',
          'No accuracy percentage is reported when the comparable denominator is zero. Unknown completion labels are not inferred from activity names or from attack=none.',
          '',f"Label status: verified {result['label_status_counts'].get('verified',0)}; unverified {result['label_status_counts'].get('unverified',0)}. Comparable: verified {result['comparable_label_status_counts'].get('verified',0)}; unverified {result['comparable_label_status_counts'].get('unverified',0)}.",
          f"Unknown expectations: {result['unknown_expected']}; missing pipeline results: {result['missing_results']}.",'',
          '|clip_id|expected|verdict|agree?|','|---|---|---|---|']
    for row in rows:
        answer='yes' if row['agree'] is True else 'no' if row['agree'] is False else 'separate: NEEDS REVIEW' if row['verdict']=='NEEDS REVIEW' else 'not comparable'
        text.append(f"|{row['clip_id']}|{row['expected']}|{row['verdict']}|{answer}|")
    if public:text+=['','Restricted-source identifiers are omitted from this public table; the unchanged aggregate counts above include their unevaluable rows.']
    text+=['','Only data/manifest.csv TEST entries were compared. Synthetic sidecars are excluded. A goal label alone is not a complete quality/verification label. These clips include reused development footage; this is not a held-out accuracy claim. No rules were changed after reading these labels.']
    return '\n'.join(text)+'\n'

if __name__=='__main__':
    with (ROOT/'data/manifest.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
    cards={}
    for path in (ROOT/'outputs/manifest_run').glob('*.json'):
        if path.name=='batch.json':continue
        card=json.loads(path.read_text()).get('card')
        if card:cards[card['clip_id']]=card
    result=compare(rows,cards)
    out=ROOT/'outputs/final';out.mkdir(parents=True,exist_ok=True)
    (out/'agreement.json').write_text(json.dumps(result,indent=2))
    (out/'agreement.md').write_text(markdown(result))
    print(markdown(result))
