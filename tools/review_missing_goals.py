"""User-invoked missing-goal recovery. Without --execute, only list pending clip IDs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
VALID_GOALS={'yes','no','unsure'}


def missing_results(folder: Path) -> list[Path]:
    found=[]
    for path in sorted(folder.glob('*.json')):
        card=json.loads(path.read_text()).get('card')
        if card and (card.get('claude_review') or {}).get('goal',{}).get('verdict') not in VALID_GOALS:
            found.append(path)
    return found


def main(execute: bool=False):
    paths=missing_results(ROOT/'outputs/manifest_run')
    print('Missing cached goal:',', '.join(p.stem for p in paths) or 'none',flush=True)
    if not execute:return
    import pandas as pd
    from app.service import evaluate
    from truetake.contracts import load_contract
    from truetake.detect_claude import detect_claude,MODEL
    from truetake.evidence import select_evidence,save_selected_frames
    from truetake.export import export_clip
    from truetake.ingest import load_recording
    from truetake.licenses import apply_license
    from truetake.report import build_card,write_json
    for path in paths:
        saved=json.loads(path.read_text());old=saved['card'];name=old['clip_id']
        contract=load_contract(old.get('task_contract') or str(ROOT/'contracts/mercor_folding.yaml'))
        rec=replace(load_recording(old['video_path'],old.get('sensors_path')),clip_id=name)
        table=pd.DataFrame(json.loads(Path(old['table_path']).read_text()));table.at[0,'contract']=contract
        folder=ROOT/'outputs/manual_goal_reviews'/name;folder.mkdir(parents=True,exist_ok=True)
        table=save_selected_frames(rec,select_evidence(table),str(folder))
        table=detect_claude(table,str(folder/'claude_cache'),str(ROOT/'prompts/review_focused_v1.md'),MODEL,
                            rec=rec,evidence_dir=str(folder))
        if (table.iloc[0].get('claude_review') or {}).get('goal',{}).get('verdict') not in VALID_GOALS:
            raise RuntimeError('No valid goal returned for '+name+'; saved results were not replaced')
        checks,quality=evaluate(table,contract)
        card=apply_license({**old,**build_card(name,table,checks,quality)},contract)
        card['export']=export_clip(rec,card,str(folder/'exports'),table)
        card['table_path']=str(folder/'frames.json');card['card_path']=str(folder/'card.json')
        table.to_json(card['table_path'],orient='records',double_precision=15)
        write_json(card['card_path'],card);saved['card']=card;write_json(path,saved)
        print(name,card.get('verdict'),flush=True)
    subprocess.run(['uv','run','python','tools/build_demo_report.py'],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Allow paid calls for missing goals only, then rebuild reports')
    main(parser.parse_args().execute)
