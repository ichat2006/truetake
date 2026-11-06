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
