from truetake.verdicts import apply_verdict
from truetake.quality import check_quality
from truetake.models import CheckResult
from test_lab_quality import lab_table

def test_measured_failure_overrides_a_positive_goal():
    table=lab_table()
    table['hand_count']=0.
    checks,quality=check_quality(table)
    checks,result=apply_verdict(table,checks,quality)
    assert result['verdict']=='REJECTED'
    assert '0%' in result['decision_reasons'][0]

def test_gyro_failure_wins_over_close_calls():
    table=lab_table()
    checks,quality=check_quality(table)
    checks.append(CheckResult('verification.gyro',False,.1,'mismatch',[]))
    _,result=apply_verdict(table,checks,quality)
    assert result['verdict']=='REJECTED'

def test_goal_unsure_cannot_be_approved():
    table=lab_table()
    table.at[0,'claude_review']['goal']['verdict']='unsure'
    checks,quality=check_quality(table)
    _,result=apply_verdict(table,checks,quality)
    assert result['verdict']=='NEEDS REVIEW'

def test_all_failures_and_reviews_are_preserved_and_cached_goal_is_unchanged():
    from copy import deepcopy
    table=lab_table()
    table['hand_count']=0.
    table.at[0,'claude_review']['goal']['verdict']='unsure'
    original=deepcopy(table.iloc[0].claude_review)
    checks,quality=check_quality(table)
    _,result=apply_verdict(table,checks,quality)
    assert result['verdict']=='REJECTED'
    assert result['failure_reasons'] and result['review_reasons']
    assert table.iloc[0].claude_review==original

def test_gyro_first_reason_links_to_its_own_timestamp():
    table=lab_table()
    checks,quality=check_quality(table)
    checks.append(CheckResult('verification.gyro',False,.1,'mismatch',[]))
    _,result=apply_verdict(table,checks,quality)
    assert result['decision_timestamp_s']==0.
