from truetake.verdicts import apply_verdict
from truetake.quality import check_quality
from truetake.models import CheckResult
from test_lab_quality import lab_table

def test_measured_failure_overrides_a_positive_goal():
    table=lab_table()
    table['hands_inside']=0.
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
    assert result['verdict'] in ('NEEDS REVIEW','REJECTED')
