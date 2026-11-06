"""Report washing checklist evidence; only train rows influence common-rule warnings."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def checklist(cards: list[dict]) -> tuple[list[dict],list[str]]:
    rows=[]
    for card in cards:
        manifest=card.get('manifest_entry',{})
        if card.get('source')=='variant' or card.get('fixture_kind'):
            continue
        if not Path(card.get('task_contract','')).name=='wash_dishes.yaml':
            continue
        for rule in (card.get('measured') or {}).get('rules',[]):
            rows.append({'clip_id':card['clip_id'],'split':card.get('split'),
                'task_expected':manifest.get('task_expected'), 'verdict':card.get('verdict',card['status']),
                'rule':rule['rule_id'],'coverage':rule['value'],'target':rule['target'],
                'state':rule['state'],'observed_fraction':rule['observed_fraction'],
                'goal':(card.get('claude_review') or {}).get('goal',{}).get('verdict','unavailable')})
    train=[row for row in rows if row['split']=='train' and row['task_expected']=='washing']
    warnings=[]
    for rule in sorted({row['rule'] for row in train}):
        relevant=[row for row in train if row['rule']==rule]
        bad=[row for row in relevant if row['state']!='pass']
        if len(bad)>len(relevant)/2:
            examples='; '.join(f"{row['clip_id']}: {row['coverage']:.1%} vs {row['target']:.0%} ({row['state']})"
                              if row['coverage'] is not None else f"{row['clip_id']}: insufficient observations"
                              for row in bad[:3])
            warnings.append(f'{rule}: {len(bad)}/{len(relevant)} real washing TRAIN clips fail or need review. {examples}')
    return rows,warnings

if __name__=='__main__':
    cards=[]
    for path in (ROOT/'outputs/manifest_run').glob('*.json'):
        if path.name=='batch.json':continue
        card=json.loads(path.read_text()).get('card')
        if card:cards.append(card)
    rows,warnings=checklist(cards)
    out=ROOT/'outputs/overnight'
    with (out/'step6_washing.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/'step6_washing.json').write_text(json.dumps({'rows':rows,'warnings':warnings},indent=2))
    report=out/'REPORT.md'
    report.write_text(report.read_text()+'\n## Step 6 — washing checklist audit\n\n'+
        '\n'.join(warnings)+'\n\nTargets unchanged. See step6_washing.csv for all coverage/target/goal values, including test rows for reporting only. Coverage currently uses the clip minus compiled exclusions as a task-time proxy; action-only segmentation is not established.\n')
    print(json.dumps({'rows':rows,'warnings':warnings},indent=2))
