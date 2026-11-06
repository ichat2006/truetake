from copy import deepcopy
from pathlib import Path
import subprocess
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import service as s

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'UPLOADS',tmp_path/'outputs/uploads')
    monkeypatch.setattr(s,'CACHE',tmp_path/'outputs/cache')
    monkeypatch.setattr(s,'RUNS',{})
    with TestClient(app) as value:yield value

@pytest.mark.skipif(not Path('outputs/manifest_run').exists(),reason='Local demo corpus required')
def test_config_and_licence_cleared_demo(client):
    config=client.get('/api/config').json()
    assert config['example']['schema_version']==2
    demo=client.get('/api/demo').json()
    assert len(demo['recommended'])>=5
    assert 'splice_P03_109_P04_104' in demo['recommended']
    assert not any(c['source']=='hub' or c['clip_id']=='basic_fold_1_black_gap' for c in demo['clips'])
    assert client.post('/api/demo',json={'clip_ids':['basic_fold_1_black_gap']}).status_code==403
    run=client.post('/api/demo',json={}).json()
    assert client.patch(f"/api/runs/{run['id']}/clips/{run['clips'][0]['clip_id']}",json={'license':'own'}).status_code==409

def test_upload_pairing_unknown_display_and_traversal(client,tmp_path,monkeypatch):
    monkeypatch.setattr(s,'catalog',lambda:{})
    path=tmp_path/'tiny.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=5:duration=1',
        '-c:v','libx264','-threads','1',str(path)],check=True,capture_output=True)
    result=client.post('/api/uploads',files=[('files',('my clip_sensors.csv',b't,gx,gy,gz,ax,ay,az\n0,0,0,0,0,0,9.8\n','text/csv')),
        ('files',('my clip.mp4',path.read_bytes(),'video/mp4'))])
    assert result.status_code==200
    run=result.json();clip=run['clips'][0]
    assert clip['sensor_present'] and clip['license']=='unknown'
    rid=run['id'];cid=clip['clip_id']
    assert client.get(f'/api/runs/{rid}/clips/{cid}/original').status_code==403
    assert client.patch(f'/api/runs/{rid}/clips/{cid}',json={'license':'own'}).status_code==200
    assert s.RUNS[rid]['clips'][0]['source']=='own'
    assert client.get(f'/api/runs/{rid}/clips/{cid}/original').status_code==200
    thumbnail=client.get(f'/api/runs/{rid}/clips/{cid}/thumbnail')
    assert thumbnail.status_code==200
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(thumbnail.content)).width==320
    assert all(s.UPLOADS in Path(c['video_path']).parents for c in s.RUNS[rid]['clips'])
    assert client.get(f'/runs/{rid}/%2e%2e/contract.yaml').status_code==404

def test_sensor_mismatch_flag_survives_upload_cache_lookup():
    card={'clip_id':'known'}
    item={'cached_card':card,'cached_imu_matches':False}
    assert s.match_cache(item)==(card,False)

def test_byte_identical_dataset_upload_cannot_claim_own_licence(tmp_path,monkeypatch):
    source=tmp_path/'catalog.mp4';source.write_bytes(b'dataset fixture')
    uploaded=tmp_path/'renamed.mp4';uploaded.write_bytes(source.read_bytes())
    card={'video_path':str(source),'license':'CC-BY-NC-ND','source':'egodex','media_modified':True}
    monkeypatch.setattr(s,'CACHE',tmp_path/'cache');monkeypatch.setattr(s,'catalog',lambda:{'known':card})
    item={'video_path':str(uploaded),'license':'own','source':'uploaded'}
    cached,matching=s.match_cache(item)
    assert cached==card and matching
    assert item['license']=='CC-BY-NC-ND' and item['license_locked']
    from truetake.licenses import apply_license
    assert not apply_license(item)['public_media_allowed']

def test_export_reuse_requires_identical_plan_input_and_artifact_hashes(tmp_path):
    from dataclasses import replace
    from test_edit_plan import recording,export_card
    from truetake.edit_plan import build_edit_plan
    original=tmp_path/'original.mp4';original.write_bytes(b'original fixture')
    output=tmp_path/'retained.mp4';output.write_bytes(b'encoded fixture')
    rec=replace(recording(),video_path=str(original));card=export_card(rec)
    plan=build_edit_plan(rec,card)
    cached={'export':{'video_path':str(output),'source_sha256':s.digest(original),
        'artifact_sha256':{output.name:s.digest(output)},'plan':{k:v for k,v in plan.items() if k not in {'source_frame_indices','expected_frame_times'}}}}
    assert s.reuse_export(rec,card,None,cached,True)['cached']
    cached['export']['plan']['output_duration_s']+=1e-12
    assert s.reuse_export(rec,card,None,cached,True)['cached']
    cached['export']['plan']['output_duration_s']+=.01
    assert s.reuse_export(rec,card,None,cached,True) is None
    cached['export']['plan']['output_duration_s']=plan['output_duration_s']
    assert s.reuse_export(rec,card,None,cached,False) is None
    output.write_bytes(b'corrupted')
    assert s.reuse_export(rec,card,None,cached,True) is None

def test_review_signature_changes_for_goal_but_not_licence_or_targets(client):
    contract=client.get('/api/config').json()['example'];other=deepcopy(contract)
    other['intended_use']='commercial_training'
    other['process'][1]['target']=.99
    assert s.review_signature(contract)==s.review_signature(other)
    other['goal']['description']='Walk to another room.'
    assert s.review_signature(contract)!=s.review_signature(other)

def test_examine_viewer_is_pinned_and_unknown_clips_are_rejected(client):
    from importlib.metadata import version
    assert client.get('/api/config').json()['rerun_version']==version('rerun-sdk')
    assert client.get('/viewer/0.0.1/index.html').status_code==404
    assert client.get('/api/examine/not-a-run/missing').status_code==404

def test_examine_preserves_credit_and_labels_static_original_preview(client,tmp_path,monkeypatch):
    import app.main as main
    timeline=tmp_path/'timeline.rrd';timeline.with_suffix('.preview.jpg').write_bytes(b'thumbnail fixture')
    monkeypatch.setattr(main,'ensure_recording',lambda run,item:timeline)
    run=s.create_run([{'clip_id':'original','license':'CC-BY-NC-ND'}])
    result=client.get(f"/api/examine/{run['id']}/original").json()
    assert result['credit_line']=='EgoDex, Apple — CC BY-NC-ND'
    assert result['preview_kind']=='static'

def test_savings_endpoint_recomputes_assumptions_without_changing_verdicts(client):
    run=s.create_run([{'clip_id':'clip_'+str(i),'license':'own'} for i in range(6)])
    run['counts']={'APPROVED':1,'REJECTED':1,'NEEDS REVIEW':3,'DUPLICATE':1}
    result=client.get(f"/api/savings/{run['id']}").json()
    assert result['percent_saved']==50 and result['net_saved_usd']==2.5
    assert client.get(f"/api/savings/{run['id']}?reviewer_rate=50").json()['net_saved_usd']==5
    assert run['counts']=={'APPROVED':1,'REJECTED':1,'NEEDS REVIEW':3,'DUPLICATE':1}
    assert client.get(f"/api/savings/{run['id']}?minutes_per_clip=-1").status_code==422

def test_processing_error_is_a_visible_needs_review_result(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'UPLOADS',tmp_path/'uploads')
    monkeypatch.setattr(s,'process_clip',lambda *args:(_ for _ in ()).throw(ValueError('Unreadable video fixture')))
    run=s.create_run([{'clip_id':'broken','name':'broken.mp4','video_path':str(tmp_path/'missing.mp4'),'license':'unknown'}])
    from truetake.contracts import load_contract
    contract=load_contract('contracts/mercor_folding.yaml')
    s.run_job(run,contract)
    assert run['counts']=={'NEEDS REVIEW':1}
    assert run['clips'][0]['verdict']=='NEEDS REVIEW'
    assert 'Unreadable video fixture' in run['clips'][0]['error']
    assert run['status']=='complete'
