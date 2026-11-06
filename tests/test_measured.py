from pathlib import Path
import pytest
import yaml
from truetake.measured import band_result, context
from test_lab_quality import lab_table

@pytest.mark.parametrize("target,lower,upper",[(.7,.6,.8),(.9,.8,.95),(1.,.9,1.)])
def test_target_band_formula(target,lower,upper):
    result=band_result(target,target,.1,.95)
    assert result["lower"] == pytest.approx(lower)
    assert result["upper"] == pytest.approx(upper)
    assert band_result(lower-.001,target,.1,.95)["state"] == "fail"
    assert band_result(lower,target,.1,.95)["state"] == "review"
    assert band_result(upper,target,.1,.95)["state"] == "pass"

def test_defaults_and_overrides():
    table=lab_table()
    contract=table.iloc[0].contract.copy()
    contract["process"]=[{"id":"hands","measurement":"hands_inside","target":.7}]
    base=context(table,contract)["rules"][0]
    from truetake.measured import defaults
    assert base["band"] == defaults()["review_band"]
    contract["process"][0]["review_band"]=.03
    assert context(table,contract)["rules"][0]["band"] == .03

def test_seconds_use_seconds_band_without_fractional_cap():
    assert band_result(11.,10.,3.)["state"] == "review"
    assert band_result(13.,10.,3.)["state"] == "pass"

def test_one_hand_counts_without_changing_hands_inside():
    import pandas as pd
    table=pd.DataFrame({'t':[0.,.1,.2,.3,.4,.5], 'sample_end':[.1,.2,.3,.4,.5,.6],
        'people_status':['ok','not_sampled','ok','not_sampled','ok','not_sampled'],
        'hand_count':[1.,None,0.,None,2.,None], 'hands_inside':[0.,None,0.,None,1.,None]})
    contract={'process':[{'id':'inside','measurement':'hands_inside','target':.6},
                         {'id':'present','measurement':'hand_present','target':.6,'review_band':.1}]}
    inside,present=context(table,contract)['rules']
    assert inside['value']==pytest.approx(1/3)
    assert present['value']==pytest.approx(2/3)
    assert present['state']=='review'
    assert present['observed_fraction']==pytest.approx(1.)

def test_hand_presence_does_not_hold_past_limit_or_through_errors():
    import pandas as pd
    table=pd.DataFrame({'t':[0.,.1,.2,.3,.4,.5], 'sample_end':[.1,.2,.3,.4,.5,.6],
        'people_status':['ok','not_sampled','not_sampled','not_sampled','error','not_sampled'],
        'hand_count':[1.,None,None,None,None,None]})
    rule=context(table,{'process':[{'id':'one','measurement':'hand_present','target':.6}]})['rules'][0]
    assert rule['observed_fraction']==pytest.approx(.5)
    assert rule['value']==1.

def test_missing_hand_count_is_unknown_even_with_ok_status():
    import pandas as pd
    table=pd.DataFrame({'t':[0.,.1], 'sample_end':[.1,.2], 'people_status':['ok','not_sampled'],
                        'hand_count':[None,None]})
    rule=context(table,{'process':[{'id':'one','measurement':'hand_present','target':.6}]})['rules'][0]
    assert rule['value'] is None and rule['observed_fraction']==0.
