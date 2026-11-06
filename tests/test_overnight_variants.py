import importlib.util
from pathlib import Path

def test_insertion_preserves_tail_and_uses_correct_padding():
    spec=importlib.util.spec_from_file_location('variants',Path('tools/overnight_variants.py'))
    import sys
    sys.path.insert(0,str(Path('tools').resolve()))
    try:
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        black=module.insertion_filter('black_gap',5.)
        freeze=module.insertion_filter('freeze',6.)
        assert 'stop_mode=add:stop_duration=5.0:color=black' in black
        assert 'stop_mode=clone:stop_duration=6.0' in freeze
        assert 'trim=start=8.0,setpts=PTS-STARTPTS[tail]' in black
        assert 'concat=n=2:v=1:a=0' in black
    finally:
        sys.path.pop(0)

def test_internal_black_gap_compiles_but_final_still_requires_goal_and_targets():
    import numpy as np
    import pandas as pd
    from truetake.contracts import load_contract
    from truetake.lab_quality import check_lab_quality
    t=np.arange(0,20,.1)
    table=pd.DataFrame({'t':t,'sample_end':t+.1,'sensor_present':False,
                        'brightness':np.where((t>=8)&(t<13),0.,100.),'frame_change':.1})
    table['contract']=pd.Series([load_contract('contracts/mercor_folding.yaml')]+[None]*(len(t)-1),dtype=object)
    table['clip_meta']=pd.Series([{'display_width':1920,'display_height':1080,'fps':30}]+[None]*(len(t)-1),dtype=object)
    checks,result=check_lab_quality(table)
    assert np.allclose(result['excluded_intervals'],[[9.,12.]])
    assert result['status']=='REVIEW' # a local cut does not confirm the goal
    assert any(check.name=='process.local_unusable_time' for check in checks)
