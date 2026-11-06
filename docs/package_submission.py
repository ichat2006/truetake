"""Build the small offline public demo from saved report assets, without inference."""
from copy import deepcopy
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote,urlsplit
import zipfile

from truetake.presentation import render_report

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/submission_demo'
FINAL=ROOT/'outputs/final'
THREADS=2
VIDEO_HEIGHT=480
VIDEO_CRF=30
TIMEOUT_S=600
MAX_DEMO_BYTES=100_000_000
def forbidden_identifiers():
    from truetake.licenses import catalog_metadata,apply_license
    return {name for name,meta in catalog_metadata().items() if not apply_license(meta)['public_media_allowed']}

def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()

def compact_video(source,target,cache):
    signature={'source_sha256':digest(source),'height':VIDEO_HEIGHT,'crf':VIDEO_CRF,'encoder':'libx264','pix_fmt':'yuv420p','faststart':True,'audio':False}
    key=str(target.relative_to(DEST))
    if target.is_file() and cache.get(key,{}).get('signature')==signature and cache[key].get('sha256')==digest(target):return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.stem+'.encoding.mp4')
    print('ENCODE',key,flush=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads',str(THREADS),'-i',str(source),
        '-filter_threads','1','-vf',f'scale=-2:{VIDEO_HEIGHT}','-an','-map_metadata','-1',
        '-c:v','libx264','-threads',str(THREADS),'-preset','veryfast','-crf',str(VIDEO_CRF),
        '-pix_fmt','yuv420p','-fps_mode','passthrough','-movflags','+faststart',str(temporary)],check=True,timeout=TIMEOUT_S)
    temporary.replace(target)
    probe=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,height,pix_fmt','-of','json',str(target)])
    stream=json.loads(probe)['streams'][0]
    if stream['codec_name']!='h264' or stream['pix_fmt']!='yuv420p' or stream['height']!=VIDEO_HEIGHT:
        raise ValueError('Unexpected compact-video encoding')
    cache[key]={'signature':signature,'sha256':digest(target),'encoding':stream}
    (FINAL/'encode_cache.json').write_text(json.dumps(cache,indent=2))

def finalize_html(text,selected):
    def keep_video(match):
        return match[0] if html.unescape(match[1]) in selected else "<div class='empty-player'>Video omitted from this lightweight bundle; thumbnail and reasons retained.</div>"
    text=re.sub(r"<video\b[^>]*\bid='(?:video_|corrected_)([^']+)'[^>]*>\s*</video>",keep_video,text)
    return re.sub(r"<p><a href='[^']*'>Rerun recording</a>.*?</p>",
        '<p>Large recordings and per-frame tables are omitted from this offline bundle.</p>',text,flags=re.S)

class LinkAudit(HTMLParser):
    def __init__(self,root):
        super().__init__();self.root=Path(root).resolve();self.anchors=set();self.targets=set();self.videos=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.anchors.add(attrs['id'])
        if tag=='video':self.videos.append(attrs.get('src',''))
        for key in ['src','href']:
            value=attrs.get(key)
            if not value:continue
            if value.startswith('#'):self.targets.add(value[1:]);continue
            url=urlsplit(value)
            if url.scheme in {'http','https'} and tag=='a':continue
            if url.scheme or url.netloc:raise ValueError('Nonlocal media/link in offline bundle')
            target=(self.root/unquote(url.path)).resolve()
            if self.root not in target.parents or not target.is_file():raise ValueError('Missing or escaping bundle asset: '+value)
    def validate(self,text):
        self.feed(text)
        if not self.targets<=self.anchors:raise ValueError('Broken card anchor')
        return self.videos

def build():
    FINAL.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
        # Preserve prior generated output instead of deleting it or mixing stale assets.
        import uuid
        previous=FINAL/('previous_submission_demo_'+uuid.uuid4().hex[:8])
        DEST.rename(previous)
    DEST.mkdir(parents=True)
    batch=json.loads((ROOT/'outputs/demo/batch.json').read_text())
    from truetake.licenses import apply_license
    cards=deepcopy({name:card for name,card in batch['cards'].items() if apply_license(card)['public_media_allowed']})
    forbidden=forbidden_identifiers()
    stops=batch['analysis']['start_here']
    selected={item['clip_id'] for item in stops}
    if len(selected)!=5 or not selected<=set(cards):raise ValueError('Expected five public Start-here clips')
    cache_path=FINAL/'encode_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    videos=[]
    for name,card in cards.items():
        for index,item in enumerate(card.get('thumbnails',[])):
            target=DEST/'thumbnails'/name/(f'{index:04d}'+Path(item['path']).suffix)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item['path'],target)
            item.clear();item.update(t=float(batch['cards'][name]['thumbnails'][index]['t']),path=str(target))
        original=batch['cards'][name]
        if name in selected:
            target=DEST/'media'/(name+'_original.mp4')
            if card.get('license_permissions',{}).get('modify',False):compact_video(original['preview_path'],target,cache)
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original['preview_path'],target)
            card['preview_path']=str(target);videos.append(str(target.relative_to(DEST)))
        else:card['preview_path']=card['thumbnails'][0]['path'] if card.get('thumbnails') else str(DEST/'omitted')
        old_export=original.get('export') or {}
        card['export']={k:v for k,v in old_export.items() if k in {'export_status','eligible_for_training','plan','output_duration_s','audio_policy','sensor_policy','post_edit_task_assessment','reason'}}
        if name in selected and old_export.get('video_path') and card.get('license_permissions',{}).get('modify',False):
            suffix='_trimmed.mp4' if old_export.get('plan',{}).get('applied_cuts') else '_retained.mp4'
            target=DEST/'media'/(name+suffix)
            compact_video(old_export['video_path'],target,cache)
            card['export']['video_path']=str(target);videos.append(str(target.relative_to(DEST)))
        entry=original.get('manifest_entry') or {}
        card['manifest_entry']={k:entry.get(k) for k in ['source','license','label_status']}
    analysis={k:deepcopy(batch['analysis'].get(k)) for k in ['map','scene_distinctness','film_next','value_ranking','start_here']}
    text=finalize_html(render_report(cards,analysis,DEST),selected)
    if any(name in text for name in forbidden):raise ValueError('Restricted-source identifier in public HTML')
    (DEST/'index.html').write_text(text)
    from final_agreement import markdown
    agreement=json.loads((FINAL/'agreement.json').read_text())
    (DEST/'agreement.md').write_text(markdown(agreement,public=True))
    for suffix in ['csv','png']:
        shutil.copyfile(ROOT/f'outputs/gyro_swap_eval_fixed.{suffix}',DEST/f'gyro_swap_eval_fixed.{suffix}')
    shutil.copyfile(ROOT/'outputs/demo/motion/splice_P03_109_P04_104.svg',DEST/'splice_window_plot.svg')
    (DEST/'README_DEMO.md').write_text('''# TrueTake offline demo

Unzip the complete folder, then double-click index.html in Chrome or Edge. No server, credentials or network connection is required. Start-here tiles jump to five actual saved results. Only those five cards include video; all others retain thumbnails and reasons.

Where modification is permitted, media are 480p, CRF 30, H.264/yuv420p, faststart and silent. Original-only media are copied byte-for-byte, with no trim, blur or re-encoding. No unlicensed variants are included. The tiles state the actual saved outcomes, not a promised cut or rejection. Duplicate footage is a controlled alias, not a second independent recording. Large per-frame tables and Rerun files are intentionally omitted.

agreement.md retains the full aggregate counts but omits restricted-source identifiers. Completion labels are unknown, so 0/0 is not an accuracy claim. Gyro plots/CSV are TRAIN development evidence from seven pairs with one fixed mapping, not test accuracy. The splice window plot is evidence only.

## Sources and licenses

EPIC-KITCHENS-100: https://epic-kitchens.github.io/2024 — CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ . Credit Dima Damen et al., Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100, IJCV 130, 33–55 (2022), https://doi.org/10.1007/s11263-021-01531-2 . Demo media are resized silent excerpts; the splice joins two source windows. These modifications are not original unedited footage. No endorsement is implied; original noncommercial terms remain applicable.

EgoDex: https://github.com/apple-aiml-research/ml-egodex — dataset terms CC-by-NC-ND. Credit Ryan Hoque et al., EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video (2025), https://arxiv.org/abs/2505.11709 . The corrected source is egodex, not own. Only original video bytes are included; no EgoDex blur/trim/variant exports are permitted. Source credit: EgoDex, Apple — CC BY-NC-ND.

No restricted-source media, thumbnails or identifiers are included. Project code does not relicense any dataset asset.
''')
    manifest={'start_here':stops,'video_files':videos,'encoding':{'height':VIDEO_HEIGHT,'crf':VIDEO_CRF,'codec':'h264','pixel_format':'yuv420p','audio':False},
              'notes':['Only five Start-here cards have video.','Pending derivatives are not accepted training footage.']}
    (DEST/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    references=LinkAudit(DEST).validate(text)
    if len(references)!=len(videos):raise ValueError('Video reference count differs from allowed media')
    prefix='sk'+'-ant'
    for path in DEST.rglob('*'):
        if not path.is_file():continue
        if any(name in str(path.relative_to(DEST)) for name in forbidden) or path.name.startswith('.env') or 'data' in path.relative_to(DEST).parts:
            raise ValueError('Forbidden bundle path')
        if path.suffix in {'.html','.md','.json','.csv','.svg'}:
            contents=path.read_text()
            if any(name in contents for name in forbidden) or prefix in contents:raise ValueError('Forbidden identifier or credential prefix in bundle text')
    archive=FINAL/'truetake_demo.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():output.write(path,Path('submission_demo')/path.relative_to(DEST))
    size=archive.stat().st_size
    print(f'DEMO ZIP: {size} bytes ({size/1e6:.2f} MB)',flush=True)
    if size>MAX_DEMO_BYTES:raise SystemExit('STOP: demo zip exceeds 100 MB; not copied for submission')
    home=Path.home()/'truetake_demo.zip';shutil.copyfile(archive,home)
    result={'demo_zip':str(home),'demo_bytes':size,'start_here':[item['clip_id'] for item in stops],
            'video_count':len(videos),'local_references_valid':True,'restricted_source_matches':0,
            'agreement_line':agreement['line'],'limitations':[item['limitation'] for item in stops if item.get('limitation')]}
    (FINAL/'package_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':build()
