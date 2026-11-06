"""FastAPI endpoints for the local TrueTake single-page app."""
from pathlib import Path
import json
import re
import threading

from fastapi import FastAPI,File,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse
import yaml

from app import service as s
from truetake.licenses import apply_license,policy,resolve_cached_path
from truetake.contracts import load_contract

app=FastAPI(title='TrueTake',docs_url='/api/docs')

def run_or_404(rid):
    if rid not in s.RUNS:raise HTTPException(404,'Run not found')
    return s.RUNS[rid]

@app.get('/')
def index():return FileResponse(Path(__file__).with_name('index.html'))

@app.get('/api/config')
def config():
    return {'example':load_contract(str(s.ROOT/'contracts/mercor_folding.yaml')),'policy':policy(),'stages':s.STAGES}

@app.post('/api/preview')
async def preview(request:Request):
    contract=await request.json()
    return {'yaml':yaml.safe_dump(contract,sort_keys=False,allow_unicode=True)}

@app.get('/api/demo')
def demo():
    cards=s.demo_catalog()
    return {'clips':[s.public_clip({**card,'name':name,'license_locked':True}) for name,card in cards.items()],
            'recommended':s.recommended(cards)}

@app.post('/api/demo')
async def load_demo(request:Request):
    body=await request.json();cards=s.demo_catalog();ids=body.get('clip_ids') or s.recommended(cards)
    if not ids:raise HTTPException(422,'No licensed demo clips installed; upload footage instead')
    if any(name not in cards for name in ids):raise HTTPException(403,'Demo source is not public-shareable')
    clips=[{**cards[name],'name':name,'cached_card':cards[name],'cached_imu_matches':True,'license_locked':True,'known_origin':True} for name in ids]
    try:return s.view(s.create_run(clips,True))
    except ValueError as error:raise HTTPException(422,str(error))

@app.post('/api/uploads')
async def upload(files:list[UploadFile]=File(...)):
    rid=__import__('uuid').uuid4().hex;folder=s.UPLOADS/rid/'raw';folder.mkdir(parents=True)
    saved={};videos=[]
    for file in files:
        name=Path(file.filename.replace('\\','/')).name
        stem=re.sub('[^a-zA-Z0-9_-]','_',Path(name).stem)[:80]
        suffix=Path(name).suffix.lower()
        if suffix not in {'.mp4','.mov','.mkv','.webm','.avi','.csv'}:raise HTTPException(422,'Use video files and matching *_sensors.csv')
        key=stem+suffix
        if key in saved:raise HTTPException(422,'Duplicate upload name')
        path=folder/key;size=0
        with path.open('xb') as handle:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>s.MAX_UPLOAD_BYTES:raise HTTPException(413,'File exceeds 512 MB')
                handle.write(chunk)
        saved[key]=path
        if suffix!='.csv':videos.append((stem,name,path))
    clips=[]
    for stem,name,path in videos:
        if any(c['clip_id']==stem for c in clips):raise HTTPException(422,'Video base names must be unique')
        sensor={name.casefold():path for name,path in saved.items()}.get((stem+'_sensors.csv').casefold())
        duration=0.
        try:
            from truetake.ingest import load_recording
            duration=load_recording(str(path)).meta['duration_s']
        except (ValueError,OSError,RuntimeError):pass
        item={'clip_id':stem,'name':name,'video_path':str(path),'sensors_path':str(sensor) if sensor else None,
            'sensor_present':sensor is not None,'duration_s':duration,'license':policy()['default_license'],'source':'uploaded'}
        cached,matching=s.match_cache(item)
        if cached:item.update(cached_card=cached,cached_imu_matches=matching)
        clips.append(item)
    if not clips:raise HTTPException(422,'Add at least one video')
    try:run=s.create_run(clips)
    except ValueError as error:raise HTTPException(422,str(error))
    # Raw files already live below outputs/uploads/<upload-id>/; retain that ID.
    old_id=run['id'];s.RUNS.pop(old_id);run.update(id=rid,folder=str(folder.parent));s.RUNS[rid]=run
    return s.view(run)

@app.get('/api/runs/{rid}')
def state(rid:str):
    run=run_or_404(rid);result=s.view(run);result['eligible_commercial']=run.get('eligible_commercial',0);return result

@app.patch('/api/runs/{rid}/clips/{cid}')
async def licence(rid:str,cid:str,request:Request):
    run=run_or_404(rid)
    if run['status']!='ready':raise HTTPException(409,'This run has started')
    item=next((c for c in run['clips'] if c['clip_id']==cid),None)
    if not item:raise HTTPException(404,'Clip not found')
    if item.get('license_locked'):raise HTTPException(409,'Known source licence cannot be replaced')
    value=(await request.json()).get('license')
    if value not in policy()['licenses']:raise HTTPException(422,'Unknown licence')
    item['license']=value
    item['source']=policy()['licenses'][value].get('declared_source','unknown')
    return s.view(run)

@app.post('/api/runs/{rid}/start')
async def start(rid:str,request:Request):
    run=run_or_404(rid)
    if run['status']!='ready':raise HTTPException(409,'Run already started')
    contract=await request.json();path=Path(run['folder'])/'contract.yaml';path.write_text(yaml.safe_dump(contract,sort_keys=False))
    try:contract=load_contract(str(path))
    except (ValueError,TypeError,KeyError) as error:raise HTTPException(422,str(error))
    run['status']='queued'
    threading.Thread(target=s.run_job,args=(run,contract),daemon=True).start()
    return s.view(run)

@app.get('/api/runs/{rid}/clips/{cid}/{kind}')
def media(rid:str,cid:str,kind:str):
    run=run_or_404(rid);item=next((c for c in run['clips'] if c['clip_id']==cid),None)
    if not item or kind not in {'thumbnail','original'}:raise HTTPException(404)
    licensed=apply_license(item)
    if not licensed['media_display_allowed']:raise HTTPException(403,'Licence does not allow display')
    if kind=='original':return FileResponse(item['video_path'])
    card=item.get('cached_card') or {};thumbs=card.get('thumbnails',[])
    if thumbs:return FileResponse(resolve_cached_path(thumbs[len(thumbs)//2]['path']))
    target=Path(run['folder'])/(cid+'_thumbnail.jpg')
    if not target.exists():
        import subprocess
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',item['video_path'],'-frames:v','1','-vf','scale=320:-2',str(target)],check=True,timeout=30)
    return FileResponse(target)

@app.get('/runs/{rid}/{asset:path}')
def artifact(rid:str,asset:str):
    run=run_or_404(rid);root=(Path(run['folder'])/'report').resolve();path=(root/asset).resolve()
    if root not in path.parents or not path.is_file():raise HTTPException(404)
    return FileResponse(path)
