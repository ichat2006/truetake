from pathlib import Path
import subprocess

from PIL import Image

from truetake.presentation import plain_reason
from truetake.thumbnails import midpoint_thumbnail


def test_midpoint_thumbnail_is_real_320px_image(tmp_path):
    video = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=white:s=96x64:r=5:d=2', '-c:v', 'libx264', '-threads', '1', str(video)], check=True)
    thumbnail = midpoint_thumbnail(str(video), str(tmp_path / 'midpoint.jpg'), 2.)
    with Image.open(thumbnail) as image:
        assert image.width == 320
        assert image.convert('L').getextrema()[0] > 200


def test_short_clip_reason_does_not_claim_a_missing_goal():
    card = {'duration_s': 2.8, 'verdict': 'NEEDS REVIEW', 'decision_reasons': ['Goal is not confirmed']}
    assert plain_reason(card) == 'Too short to review (2.8 s)'
    card.update(duration_s=8., measured={'usable_assessment': {'target': 10.}})
    assert plain_reason(card) == 'Too short to review (8.0 s)'
    card.update(duration_s=60., measured={'usable_assessment': {'target': 10., 'value': 2.8}})
    assert plain_reason(card) == 'Too short to review (2.8 s)'


def test_example_uses_one_hand_and_plain_idle_text():
    from truetake.contracts import load_contract
    for name in ['mercor_folding', 'wash_dishes']:
        contract = load_contract(f'contracts/{name}.yaml')
        rules = {r['id']: r for r in contract['process']}
        assert rules['one_hand_visible']['measurement'] == 'hand_present'
        assert rules['idle']['description'] == 'No idle stretch over 3 s'


def test_known_insertion_shifts_goal_evidence_and_drops_sensor_claims():
    import json
    import numpy as np
    import pandas as pd
    from test_lab_quality import lab_table
    from truetake.cached_insert import transfer_insert_cache
    parent=lab_table(15)
    before=json.dumps(parent.iloc[0]['claude_review'],sort_keys=True)
    t=np.arange(0,20,.1)
    target=pd.DataFrame({'t':t,'sample_end':t+.1,'source_frame':np.arange(len(t))*3,
                         'sensor_present':False,'imu_valid':False})
    target['contract']=pd.Series([parent.iloc[0]['contract']]+[None]*(len(t)-1),dtype=object)
    target['clip_meta']=pd.Series([{}]+[None]*(len(t)-1),dtype=object)
    actual=transfer_insert_cache(parent,target)
    review=actual.iloc[0]['claude_review']
    assert review['goal']==parent.iloc[0]['claude_review']['goal']
    import pytest
    assert review['evidence'][-1]['t']==pytest.approx(19.9)
    assert not actual.sensor_present.any()
    assert actual.loc[(actual.t>=8)&(actual.t<13),'people_status'].eq('skipped_dark').all()
    assert json.dumps(parent.iloc[0]['claude_review'],sort_keys=True)==before


def test_missing_goal_recovery_selects_only_missing_reviews(tmp_path):
    import json
    from tools.review_missing_goals import missing_results
    for name,goal in [('yes','yes'),('no','no'),('unsure','unsure'),('missing',None)]:
        (tmp_path/(name+'.json')).write_text(json.dumps({'card':{'claude_review':{'goal':{'verdict':goal}}}}))
    assert [p.stem for p in missing_results(tmp_path)]==['missing']
