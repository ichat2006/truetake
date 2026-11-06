from copy import deepcopy
import pytest
from truetake.licenses import apply_license,permissions
from truetake.export import omitted_entry

@pytest.mark.parametrize('lic,modify,public,commercial',[('own',True,True,True),('CC-BY-NC-4.0',True,True,False),
    ('CC-BY-NC-ND',False,True,False),('hub-egocentric-eval',False,False,False),('unknown',False,False,False)])
def test_policy_matrix_and_independent_verdict(lic,modify,public,commercial):
    original={'clip_id':'x','status':'ACCEPT','verdict':'APPROVED','license':lic,'decision_reasons':['goal yes']}
    before=deepcopy(original);result=apply_license(original,{'intended_use':'commercial_training'})
    assert result['license_permissions']['modify']==modify
    assert result['public_media_allowed']==public
    assert result['eligibility']['eligible']==commercial
    assert result['verdict']=='APPROVED' and original==before
    if not modify:assert omitted_entry(result)['export_status']=='not_exported'

def test_original_only_cannot_share_variant_or_blur():
    card={'clip_id':'x','status':'ACCEPT','license':'CC-BY-NC-ND','media_modified':True}
    assert not apply_license(card)['public_media_allowed']
    card.update(media_modified=False,privacy_blur_intervals=[[1.,2.]])
    assert not apply_license(card)['media_display_allowed']

def test_unknown_cannot_gain_permissions():
    assert permissions('unrecognized')==permissions('unknown')

def test_showcase_selection_follows_policy_not_source_names(monkeypatch):
    import truetake.licenses as gate
    from truetake.presentation import select_start_here
    rules=gate.policy();rules['licenses']['hub-egocentric-eval'].update(share_public=True,show_original=True)
    monkeypatch.setattr(gate,'policy',lambda:rules)
    card={'clip_id':'example','status':'ACCEPT','license':'hub-egocentric-eval','source':'hub'}
    assert select_start_here({'example':card})[0]['clip_id']=='example'

def test_public_directory_policy_includes_children_but_not_sibling_private_runs():
    from truetake.licenses import is_public_directory,ROOT
    assert is_public_directory(ROOT/'outputs/demo/nested')
    assert not is_public_directory(ROOT/'outputs/uploads/run/analysis')

def test_old_no_derivatives_export_is_removed_and_proposed_cut_stays_on_original(tmp_path,monkeypatch):
    from truetake.report import write_batch_report
    original=tmp_path/'original.mp4';original.write_bytes(b'original fixture')
    card={'clip_id':'original','license':'CC-BY-NC-ND','status':'ACCEPT','verdict':'APPROVED',
        'video_path':str(original),'thumbnails':[],'checks':[],'excluded_intervals':[[1.,2.]],'duration_s':4.,
        'quality_score':100.,'usable_seconds':3.,'table_path':str(tmp_path/'frames.json'),
        'card_path':str(tmp_path/'card.json'),'rerun_path':str(tmp_path/'timeline.rrd'),
        'export':{'video_path':str(tmp_path/'forbidden_trim.mp4')}}
    monkeypatch.setattr('truetake.report.browser_preview',lambda *args:pytest.fail('No ND re-encoding'))
    output=write_batch_report({'original':card},{'map':[]},str(tmp_path))
    text=__import__('pathlib').Path(output).read_text()
    assert 'forbidden_trim.mp4' not in text and 'original.mp4' in text
    assert 'proposals only' in text and 'EgoDex, Apple' in text
    assert card['status']=='ACCEPT' and card['export']['export_status']=='not_exported'

def test_own_variant_tool_refuses_downloaded_source(tmp_path,monkeypatch):
    import csv
    from tools import make_variants_from_own as tool
    path=tmp_path/'data/manifest_v2.csv';path.parent.mkdir()
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['clip_id','source']);writer.writeheader();writer.writerow({'clip_id':'download','source':'egodex'})
    monkeypatch.setattr(tool,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='not self-recorded'):tool.create('missing.mp4','download','train')

def test_own_variants_preserve_parent_split_and_add_expected_seconds(tmp_path,monkeypatch):
    import csv,subprocess,json,hashlib
    from tools import make_variants_from_own as tool
    source=tmp_path/'own.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=5:duration=10','-c:v','libx264','-threads','1',str(source)],check=True,capture_output=True)
    row={'clip_id':'mine','source':'own','license':'own','collection':'own','participant':'self','video_path':str(source),
        'sensors_path':'','task_expected':'folding','goal_met_expected':'unknown','split':'test','attack':'none',
        'label_status':'unverified','media_modified':'false','notes':'','parent_clip_id':''}
    manifest=tmp_path/'data/manifest_v2.csv';manifest.parent.mkdir()
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    before=hashlib.sha256(source.read_bytes()).hexdigest();monkeypatch.setattr(tool,'ROOT',tmp_path)
    made=tool.create(str(source),'mine','train')
    assert all(r['source']=='own' and r['split']=='test' and not r['sensors_path'] for r in made)
    for item,expected in zip(made,[15,16]):
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',item['video_path']]))
        assert float(info['format']['duration'])==pytest.approx(expected,abs=.21)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
