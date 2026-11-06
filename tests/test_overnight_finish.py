import importlib.util
from pathlib import Path

def test_final_summary_counts_duplicates_separately_and_checks_hub():
    spec=importlib.util.spec_from_file_location('finish',Path('tools/overnight_finish.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    batch={'cards':{'a':{'status':'REJECT','source':'own','verdict':'REJECTED'},
                    'b':{'status':'ACCEPT','source':'own','fixture_kind':'controlled duplicate'}},'analysis':{}}
    result=module.artifact_summary(batch)
    assert result['verdict_counts']=={'REJECTED':1,'DUPLICATE':1}
    assert result['usable_minutes']==0 and result['hub_in_public'] is False
