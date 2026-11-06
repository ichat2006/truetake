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
