from copy import deepcopy
import importlib.util
from pathlib import Path

from test_lab_quality import lab_table

def module():
    spec=importlib.util.spec_from_file_location('cached_recompute',Path('tools/recompute_cached_demo.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def test_cached_goal_no_remains_rejected_even_when_one_hand_passes():
    table=lab_table()
    table['hand_count']=1.
    table['hands_inside']=0.
    table.at[0,'claude_review']['goal']['verdict']='no'
    review=deepcopy(table.iloc[0].claude_review)
    updated,card=module().recompute(table,{'claude_review':deepcopy(review)},table.iloc[0].contract)
    assert card['verdict']=='REJECTED'
    assert any(reason.startswith('Goal not met') for reason in card['failure_reasons'])
    assert updated.iloc[0].claude_review==review
    assert card['decision_recompute']['api_calls']==0

def test_one_hand_accepts_cached_positive_goal_without_inside_requirement():
    table=lab_table()
    table['hand_count']=1.
    table['hands_inside']=0.
    _,card=module().recompute(table,{'claude_review':deepcopy(table.iloc[0].claude_review)},table.iloc[0].contract)
    assert card['verdict']=='APPROVED'
    assert card['measured']['rules'][0]['value']==1.
    assert card['failure_reasons']==[] and card['review_reasons']==[]
