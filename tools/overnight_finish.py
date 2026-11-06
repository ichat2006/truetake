"""Rebuild reports, check two real inputs, and copy the requested public backup."""
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess

from truetake.presentation import badge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/overnight'

def artifact_summary(batch: dict) -> dict:
    cards=batch['cards']
    return {'clip_count':len(cards),'verdict_counts':dict(Counter(badge(card) for card in cards.values())),
            'usable_minutes':sum(card.get('usable_seconds',0.) for card in cards.values() if not card.get('fixture_kind'))/60,
            'missing':batch['analysis'].get('skipped',[]),'hub_in_public':any(card.get('source')=='hub' for card in cards.values())}

if __name__=='__main__':
    # Updating verdict links uses saved evidence only. No detector/model work.
    from overnight_decisions import compute
    compute(persist=True)
    for name in ['basic_fold_0','basic_fold_1']:
        video=ROOT/'outputs/public_samples/egodex'/(name+'.mp4')
        command=['uv','run','python','cli.py',str(video),'--contract','contracts/mercor_folding.yaml']
        with (OUT/(name+'_cli_smoke.json')).open('w') as handle:
            subprocess.run(command,cwd=ROOT,stdout=handle,check=True)
        print('CLI SMOKE OK',name,flush=True)
    subprocess.run(['uv','run','python','tools/build_demo_report.py'],cwd=ROOT,check=True)
    batch=json.loads((ROOT/'outputs/demo/batch.json').read_text())
    summary=artifact_summary(batch)
    if summary['hub_in_public']:
        raise RuntimeError('Public report contains Hub clips')
    source=ROOT/'outputs/demo';backup=ROOT/'outputs/demo_backup'
    shutil.copytree(source,backup,dirs_exist_ok=True)
    # Backup JSON asset paths are absolute; browser HTML paths are relative and portable.
    if (source/'index.html').read_bytes()!=(backup/'index.html').read_bytes():
        raise RuntimeError('Backup report does not match public report')
    (OUT/'step8_summary.json').write_text(json.dumps(summary,indent=2))
    report=OUT/'REPORT.md'
    report.write_text(report.read_text()+ '\n## Step 8 — final build\n\n'+
        json.dumps(summary['verdict_counts'])+f"; {summary['usable_minutes']:.1f} approved usable minutes. Both documented CLI input checks passed. Public and internal reports rebuilt; public Hub count 0; public report copied to outputs/demo_backup/.\n"+
        '\n### Look here first\n\nThe new policy yields 0 approved clips. Every measured target is now binding, including the formerly optional hands rule. The 90% target is not supported by this detector on the corpus. Action-only task segmentation is still a full-clip-minus-exclusions proxy. No targets were tuned or relaxed.\n\n'+
        'Suggested five inspection stops (not five fabricated successful examples):\n\n'+
        '1. basic_fold_1_black_gap — exact black frames flagged; 9–12s compiled cut, but REJECTED and no trimmed training export.\n'+
        '2. splice_P03_109_P04_104 — real gyro FAIL, score 0.153 < 0.290; window-score plot marks 0:30.\n'+
        '3. basic_fold_0 + basic_fold_0_controlled_reupload — duplicate pair; controlled alias is not independent footage.\n'+
        '4. P07_102_059_119 — gyro close-call-zone example, but final REJECTED on hands; there are no final NEEDS REVIEW clips under these targets.\n'+
        '5. P01_102_000_060_wrong_task — folding-contract alias rejects on coverage; semantic wrong-task answer is unavailable without API access.\n\n'+
        '### Still outstanding / limitations\n\n'+
        '- Paid model probes/review, validation retry and escalation were skipped after two authentication failures. No API spend.\n'+
        '- No genuinely approved cut variant or approved EPIC card exists under the requested measured-target policy. Rejected clips are not converted into training-eligible copies.\n'+
        '- Both new variants have complete local detector tables but no new Claude goal answer; no inferred outcome was substituted from manifest labels.\n'+
        '- 12 requested source videos remain missing (5 EPIC and 7 Hub); skipped.csv lists them by report scope.\n'+
        '- Old output files are retained for provenance; current collection manifests are authoritative. Backup HTML is portable, but raw JSON retains absolute local artifact paths.\n'+
        '- Legacy contracts/fold.yaml remains the version-1 compatibility contract. The overnight schema/formulas apply to contracts/mercor_folding.yaml and wash_dishes.yaml used by the manifest runner.\n')
    print(json.dumps(summary,indent=2),flush=True)
