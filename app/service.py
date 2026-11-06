"""Local job orchestration around the existing table checks and batch pipeline."""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import pandas as pd
import yaml

from truetake.contracts import load_contract
from truetake.licenses import apply_license,catalog_metadata,resolve_cached_path

ROOT=Path(__file__).resolve().parents[1]
UPLOADS=ROOT/'outputs/uploads'
CACHE=ROOT/'outputs/app_cache'
STAGES=['Read file','Licence','Quality','Hands','Camera motion','Phone sensor','AI review','Verdict']
RUNS={}
WORKER_LOCK=threading.Lock()
STATE_LOCK=threading.RLock()
MAX_UPLOAD_BYTES=512*1024*1024
MAX_CLIPS=24

def same_plan(a,b):
    """Ignore only timestamp serialization noise at the existing edit precision."""
    from truetake.edit_plan import TIME_TOLERANCE_S
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same_plan(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(same_plan(x,y) for x,y in zip(a,b))
    if type(a) in (int,float) and type(b) in (int,float) and (type(a) is float or type(b) is float):return abs(a-b)<=TIME_TOLERANCE_S
    return a==b

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def catalog():
    result={};metadata=catalog_metadata()
    published=ROOT/'outputs/demo/batch.json'
    public=json.loads(published.read_text()).get('cards',{}) if published.exists() else {}
    previews=CACHE/'preview_index.json'
    prepared=json.loads(previews.read_text()) if previews.exists() else {}
    for path in sorted((ROOT/'outputs/manifest_run').glob('*.json')):
        if path.name=='batch.json':continue
        card=json.loads(path.read_text()).get('card')
        if card and card['clip_id'] in metadata:
            card=apply_license({**card,**metadata[card['clip_id']]})
            shown=public.get(card['clip_id']) or prepared.get(card['clip_id'],{})
            if (same_plan(shown.get('privacy_blur_intervals'),card.get('privacy_blur_intervals'))
                    and shown.get('preview_path') and Path(shown['preview_path']).is_file()
                    and shown.get('video_path')==card.get('video_path')):
                card.update(preview_path=shown['preview_path'],presentation_preview_ready=True)
            if Path(card['video_path']).is_file():result[card['clip_id']]=card
    return result

def demo_catalog():
    return {name:card for name,card in catalog().items() if card['public_media_allowed']}

def recommended(cards):
    # Saved outcomes select the story; no expected labels are used.
    from truetake.presentation import badge,plain_reason
    chosen=[]
    predicates=[lambda n,c:badge(c)=='APPROVED' and c.get('manifest_entry',{}).get('task_expected')=='folding',
        lambda n,c:badge(c)=='NEEDS REVIEW' and (c.get('claude_review') or {}).get('goal',{}).get('verdict')=='yes' and c.get('manifest_entry',{}).get('task_expected')=='folding',
        lambda n,c:bool(c.get('fixture_kind')),
        lambda n,c:(c.get('verification') or {}).get('passed') is False,
        lambda n,c:n.endswith('_wrong_task')]
    for predicate in predicates:
        ordered=sorted(cards.items(),key=lambda pair:pair[1].get('manifest_entry',{}).get('attack')!='splice')
        found=next((name for name,card in ordered if name not in chosen and predicate(name,card)),None)
        if found:chosen.append(found)
    cut=next((name for name,c in cards.items() if c.get('demo_gap') and (c.get('export') or {}).get('video_path')),None)
    if cut:chosen.insert(0,cut)
    return chosen

def public_clip(item):
    licensed=apply_license(item)
    return {key:licensed.get(key) for key in ['clip_id','name','duration_s','license','source','credit_line','eligibility','media_display_allowed','sensor_present','stages','verdict','reason','error','cached','license_locked']}

def create_run(clips,demo=False):
    if not clips or len(clips)>MAX_CLIPS:raise ValueError('Select 1–24 clips')
    rid=uuid.uuid4().hex
    folder=UPLOADS/rid;folder.mkdir(parents=True)
    for item in clips:
        item.setdefault('name',item['clip_id']);item['stages']={stage:'waiting' for stage in STAGES}
        item.pop('verdict',None);item.pop('reason',None)
    run={'id':rid,'folder':str(folder),'clips':clips,'demo':demo,'status':'ready','counts':{},'report_url':None,'runtime_api_calls':0,'detector_runs':0}
    with STATE_LOCK:RUNS[rid]=run
    return run

def view(run):
    with STATE_LOCK:
        return {'id':run['id'],'status':run['status'],'clips':[public_clip(c) for c in run['clips']],
                'counts':dict(run['counts']),'report_url':run['report_url'],'error':run.get('error'),
                'runtime_api_calls':run['runtime_api_calls'],'detector_runs':run['detector_runs']}

def update_stage(item,stage,status='done'):
    with STATE_LOCK:item['stages'][stage]=status

def review_signature(contract):
    return {'task':contract.get('task'),'goal':contract.get('goal'),
            'process':[rule for rule in contract.get('process',[]) if rule.get('required') and not rule.get('measurement')],
            'instructions':contract.get('operator_request','')}

def match_cache(item):
    if item.get('cached_card'):return item['cached_card'],item.get('cached_imu_matches',True)
    video=Path(item['video_path']);sensor=item.get('sensors_path')
    key=digest(video)+(':'+digest(sensor) if sensor else ':none')
    index=CACHE/'index.json'
    entries=json.loads(index.read_text()) if index.exists() else {}
    if key in entries and Path(entries[key]).is_file():
        card=json.loads(Path(entries[key]).read_text())
        if card.get('known_origin'):item.update(license=card['license'],source=card['source'],license_locked=True,known_origin=True)
        return card,True
    for card in catalog().values():
        source=Path(card['video_path'])
        if source.stat().st_size==video.stat().st_size and digest(source)==key.split(':')[0]:
            old=card.get('sensors_path')
            matched=(not sensor and not old) or bool(sensor and old and digest(sensor)==digest(old))
            # Known third-party content cannot be relabelled as self-recorded.
            item.update(license=card['license'],source=card['source'],media_modified=card['media_modified'],license_locked=True,known_origin=True)
            return card,matched
    return None,False

def evaluate(table,contract):
    from truetake.quality import check_quality
    from truetake.verify_gyro import check_verification
    from truetake.verdicts import apply_verdict
    from truetake.models import CheckResult
    table.at[0,'contract']=contract
    checks,quality=check_quality(table)
    # Reuse the legacy engine's existing hard checks, rather than silently
    # ignoring hard chips that are not part of the v2 format adapter.
    extra=[name for name in ['no_faces','no_slow_motion','unedited','not_on_tripod']
           if contract.get('requirements',{}).get(name,{}).get('required')]
    if extra:
        legacy=table.copy();legacy.at[0,'contract']={'requirements':{'hard':extra},'min_usable_seconds':0}
        legacy_checks,_=check_quality(legacy)
        checks += [c for c in legacy_checks if c.name.startswith('requirements.')]
    checks += check_verification(table) if table.sensor_present.any() and 'gyro_alignment' in table else [CheckResult('verification.gyro',None,0.,'no motion data' if not table.sensor_present.any() else 'calibrated gyro profile required',[])]
    return apply_verdict(table,checks,quality)

def reuse_export(rec,card,table,cached,imu_matches):
    """Reuse actual artifact hashes and an identical edit plan, not an old verdict."""
    from truetake.edit_plan import build_edit_plan
    if not cached or not imu_matches or not apply_license(card)['license_permissions']['modify']:return None
    old=cached.get('export') or {}
    if not old.get('video_path') or not Path(old['video_path']).is_file():return None
    if old.get('source_sha256')!=digest(rec.video_path):return None
    plan=build_edit_plan(rec,card,table)
    plan={key:value for key,value in plan.items() if key not in {'source_frame_indices','expected_frame_times'}}
    if not same_plan(plan,old.get('plan')):return None
    folder=Path(old['video_path']).parent
    if not old.get('artifact_sha256') or any(not (folder/name).is_file() or digest(folder/name)!=sha for name,sha in old['artifact_sha256'].items()):return None
    return {**old,'cached':True}

def process_clip(run,item,contract):
    from truetake.ingest import load_recording
    from truetake.report import build_card
    from truetake.export import export_clip
    cached,imu_matches=match_cache(item)
    update_stage(item,'Read file','running')
    rec=replace(load_recording(item['video_path'],item.get('sensors_path')),clip_id=item['clip_id'])
    item.update(duration_s=rec.meta['duration_s'],sensor_present=rec.imu is not None)
    update_stage(item,'Read file');update_stage(item,'Licence','running')
    licensed=apply_license(item,contract)
    if not licensed['license_permissions']['analyze']:raise ValueError('Licence does not allow analysis')
    item.update(licensed);update_stage(item,'Licence')
    clip_dir=Path(run['folder'])/'analysis'/item['clip_id'];clip_dir.mkdir(parents=True,exist_ok=True)
    if cached:
        table=pd.DataFrame(json.loads(Path(cached['table_path']).read_text()))
        table['sensor_present']=rec.imu is not None
        if 'thumbnail_path' in table:
            table['thumbnail_path']=table.thumbnail_path.map(lambda p:str(resolve_cached_path(p)) if isinstance(p,str) else p)
        for stage in ['Quality','Hands','Camera motion']:update_stage(item,stage,'cached')
        update_stage(item,'Phone sensor','running')
        if not imu_matches and rec.imu is not None:
            from truetake.batch import verify_recording
            profile=(table.iloc[0].get('clip_meta') or {}).get('gyro_calibration')
            table,_=verify_recording(rec,table,profile)
        update_stage(item,'Phone sensor','cached' if imu_matches else 'done')
        old=table.iloc[0].get('contract') or {}
        compatible=review_signature(old)==review_signature(contract)
        table.at[0,'contract']=contract
        update_stage(item,'AI review','running')
        if not compatible:
            table.at[0,'claude_review']=None
            table.at[0,'claude_status']='checklist_changed_review_unavailable'
            if not run['demo']:
                from truetake.detect_claude import detect_claude
                table=detect_claude(table,str(CACHE/'claude'),str(ROOT/'prompts/review_focused_v1.md'),rec=rec,evidence_dir=str(clip_dir))
                run['runtime_api_calls']+=len(table.attrs.get('claude_call_usage',[]))
        update_stage(item,'AI review','cached' if compatible and isinstance(table.iloc[0].get('claude_review'),dict) else 'unavailable')
        checks,quality=evaluate(table,contract)
        card=build_card(item['clip_id'],table,checks,quality)
        if cached.get('fixture_kind'):card['fixture_kind']=cached['fixture_kind']
        item['cached']=True
    else:
        from truetake.batch import run_batch
        stage=Path(run['folder'])/'inputs';stage.mkdir(exist_ok=True)
        link=stage/(item['clip_id']+'.mp4')
        if not link.exists():link.symlink_to(Path(item['video_path']).resolve())
        if item.get('sensors_path'):
            sensor_link=stage/(item['clip_id']+'_sensors.csv')
            if not sensor_link.exists():sensor_link.symlink_to(Path(item['sensors_path']).resolve())
        def progress(stage,status):update_stage(item,stage,status)
        result=run_batch(str(stage),str(Path(run['folder'])/'contract.yaml'),str(Path(run['folder'])/'analysis'),
            detector_mode='onepass',render=False,export=False,clip_ids=[item['clip_id']],
            media_metadata={item['clip_id']:item},progress=progress)
        if item['clip_id'] not in result['cards']:raise ValueError(result['analysis']['errors'][0]['error'])
        card=result['cards'][item['clip_id']]
        run['runtime_api_calls']+=(card.get('claude_usage') or {}).get('actual_api_calls',0)
        run['detector_runs']+=int(not card.get('detector_cache_hit',False))
        table=pd.DataFrame(json.loads(Path(card['table_path']).read_text()))
        checks,quality=evaluate(table,contract);card.update(quality,checks=[asdict(c) for c in checks])
    card=apply_license({**card,**{k:item.get(k) for k in ['license','source','media_modified'] if k in item}},contract)
    card['known_origin']=item.get('known_origin',False)
    card['gyro_alignment']=table.iloc[0].get('gyro_alignment')
    if cached:
        for key in ['demo_gap','cache_provenance','task_contract']:
            if key in cached:card[key]=cached[key]
    if cached and cached.get('preview_path') and Path(cached['preview_path']).is_file() and same_plan(cached.get('privacy_blur_intervals'),card.get('privacy_blur_intervals')):
        card.update(preview_path=cached['preview_path'],presentation_preview_ready=True)
    card.update(video_path=str(Path(rec.video_path).resolve()),sensors_path=item.get('sensors_path'),
        card_path=str(clip_dir/'card.json'),table_path=str(clip_dir/'frames.json'),rerun_path=str(clip_dir/'timeline.rrd'))
    card['export']=reuse_export(rec,card,table,cached,imu_matches) or export_clip(rec,card,str(CACHE/'exports'),table)
    table.to_json(card['table_path'],orient='records',double_precision=15)
    Path(card['card_path']).write_text(json.dumps(card,indent=2))
    CACHE.mkdir(parents=True,exist_ok=True)
    key=digest(item['video_path'])+(':'+digest(item['sensors_path']) if item.get('sensors_path') else ':none')
    index=CACHE/'index.json';entries=json.loads(index.read_text()) if index.exists() else {};entries[key]=card['card_path'];index.write_text(json.dumps(entries))
    update_stage(item,'Verdict')
    return card

def run_job(run,contract):
    from truetake.value import analyze_batch
    from tools.build_demo_report import publish_subset
    from truetake.presentation import badge
    with WORKER_LOCK:
        run['status']='running';cards={};tables={}
        for item in run['clips']:
            try:
                clip_contract=contract
                if run.get('demo') and item.get('demo_gap'):
                    clip_contract=load_contract(item['task_contract'])
                card=process_clip(run,item,clip_contract)
            except Exception as error:
                message=re.sub('sk'+r'-ant[\w-]+','[redacted]',str(error))[:500]
                item['error']=type(error).__name__+': '+message
                for stage,state in item['stages'].items():
                    if state in {'running','waiting'}:item['stages'][stage]='error' if state=='running' else 'skipped'
                folder=Path(run['folder'])/'analysis'/item['clip_id'];folder.mkdir(parents=True,exist_ok=True)
                table=pd.DataFrame([{'t':0.,'sample_end':0.,'contract':contract,'clip_meta':{}}]);table.to_json(folder/'frames.json',orient='records')
                card=apply_license({**item,'status':'REVIEW','verdict':'NEEDS REVIEW','processing_error':item['error'],
                    'decision_reasons':[item['error']],'duration_s':item.get('duration_s',0.),'usable_seconds':0.,'quality_score':0.,
                    'thumbnails':[],'checks':[],'export':{'export_status':'not_exported','eligible_for_training':False},
                    'table_path':str(folder/'frames.json'),'card_path':str(folder/'card.json'),'rerun_path':str(folder/'timeline.rrd')},contract)
                Path(card['card_path']).write_text(json.dumps(card));update_stage(item,'Verdict','error')
            cards[item['clip_id']]=card
            tables[item['clip_id']]=pd.DataFrame(json.loads(Path(card['table_path']).read_text()))
            from truetake.presentation import plain_reason
            item.update(verdict=badge(card),reason=plain_reason(card),eligibility=card['eligibility'])
            with STATE_LOCK:
                from collections import Counter
                run['counts']=dict(Counter(badge(c) for c in cards.values()))
        try:
            analysis=analyze_batch(tables,cards)
            report=Path(run['folder'])/'report'
            publish_subset({'cards':cards,'analysis':analysis},report,False)
            rendered=json.loads((report/'batch.json').read_text())['cards']
            preview_index=CACHE/'preview_index.json';CACHE.mkdir(parents=True,exist_ok=True)
            prepared=json.loads(preview_index.read_text()) if preview_index.exists() else {}
            prepared.update({name:{key:c.get(key) for key in ['video_path','preview_path','privacy_blur_intervals']}
                             for name,c in rendered.items() if c.get('media_display_allowed')})
            preview_index.write_text(json.dumps(prepared))
            text=(report/'index.html').read_text()
            text=text.replace('</style>',":root{--muted:#686960;--ink:#1D1D1B;--line:#d9d7cf;color-scheme:light}body{background:#F4F2EC;color:#1D1D1B}.box,article,.start-tile{background:#fff;border-color:#dedbd2}h1,h2{font-family:monospace}a,.time-link{color:#0F6E6E}pre{background:#eeeae0;color:#333}.badge{background:#fff}.reason,.usable{color:#1D1D1B}.map{background:#e6e2d7}.map text{fill:#555}.empty-player{color:#555}.next{color:#0F6E6E}</style>")
            text=text.replace('</style>',"header>.brand,header>h1,header>.next{display:none}header{padding:0 0 16px}main{padding-top:24px}</style>")
            from app.examine import RERUN_VERSION
            for name,card in cards.items():
                label='Examine — needs a human' if badge(card)=='NEEDS REVIEW' else 'Examine'
                button=f"<button class='examine {'needs-human' if badge(card)=='NEEDS REVIEW' else ''}' data-examine='{name}'>{label}</button>"
                text=text.replace(f"<article id='{name}'>",f"<article id='{name}'>"+button)
            text=text.replace('</body>',"<script>document.addEventListener('click',e=>{const b=e.target.closest('[data-examine]');if(b)window.parent.postMessage({type:'truetake-examine',clip_id:b.dataset.examine},location.origin)});</script></body>")
            text=text.replace('</style>',".examine{padding:9px 15px;background:transparent;color:#0F6E6E;border:1px solid #0F6E6E;border-radius:4px;margin-bottom:12px}.examine.needs-human{background:#0F6E6E;color:white;font-weight:600}</style>")
            (report/'index.html').write_text(text)
            run['report_url']=f"/runs/{run['id']}/index.html"
            run['eligible_commercial']=sum(c['eligibility']['eligible'] and c['license_permissions']['train_commercial'] and c.get('verdict')=='APPROVED' for c in cards.values())
            run['status']='complete'
        except Exception as error:run.update(status='error',error=type(error).__name__+': '+str(error)[:300])
