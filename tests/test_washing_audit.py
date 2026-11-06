import importlib.util
from pathlib import Path

def test_test_rows_do_not_change_train_warning():
    spec=importlib.util.spec_from_file_location('audit',Path('tools/washing_audit.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def card(name,split,coverage):
        return {'clip_id':name,'source':'epic','split':split,'status':'REJECT','task_contract':'contracts/wash_dishes.yaml',
                'manifest_entry':{'task_expected':'washing'},'measured':{'rules':[{'rule_id':'hands','value':coverage,
                'target':.9,'state':'fail' if coverage<.8 else 'pass','observed_fraction':1.}]}}
    train=[card('a','train',.2),card('b','train',.3),card('c','train',.4)]
    _,warnings=module.checklist(train)
    rows,mixed=module.checklist(train+[card('test','test',1.)]*10)
    assert mixed==warnings and '3/3' in warnings[0]
    assert len(rows)==13
