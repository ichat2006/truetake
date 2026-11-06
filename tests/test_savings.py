import pytest
from app.savings import estimate


def test_demo_savings_are_assumptions_separate_from_quality():
    counts={'APPROVED':1,'REJECTED':1,'NEEDS REVIEW':3,'DUPLICATE':1}
    result=estimate(6,counts)
    assert result['automatic']==2 and result['human']==3 and result['duplicates']==1
    assert result['hours_without']==pytest.approx(.2)
    assert result['hours_with']==pytest.approx(.1)
    assert result['percent_saved']==50
    assert result['net_saved_usd']==pytest.approx(2.5)
    assert result['projection']['hours_without']==pytest.approx(1000*2/60)
    assert estimate(6,counts,reviewer_rate=50)['net_saved_usd']==pytest.approx(5.)
    assert counts=={'APPROVED':1,'REJECTED':1,'NEEDS REVIEW':3,'DUPLICATE':1}


def test_uncertain_statuses_and_unpriced_ai_are_not_free_approvals():
    result=estimate(4,{'APPROVED':1},usage={'calls':1,'cost_usd':None})
    assert result['human']==3 and result['net_saved_usd'] is None
    assert estimate(0,{})['percent_saved']==0
    assert estimate(4,{'NEEDS REVIEW':4})['percent_saved']==0
    for invalid in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):estimate(1,{},minutes_per_clip=invalid)
