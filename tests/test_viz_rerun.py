import numpy as np
import pandas as pd

from truetake.viz_rerun import aligned_motion,segment_rows

def test_segment_track_uses_only_applied_cuts():
    card={'excluded_intervals':[[1.,3.]]}
    assert segment_rows(5.,card)==[{'start':0.,'end':5.,'kind':'KEEP','reason':'Retained original footage'}]
    card['export']={'plan':{'applied_cuts':[{'start_s':1.,'end_s':3.}]}}
    segments=segment_rows(5.,card)
    assert [s['kind'] for s in segments]==['KEEP','CUT','KEEP']
    assert sum(s['end']-s['start'] for s in segments)==5.

def test_mapped_display_applies_lag_and_sign():
    t=np.arange(0,20,.1)
    f=lambda t: np.sin(2*np.pi*.7*t)
    table=pd.DataFrame({'t':t,'flow_t':t,'flow_status':'ok','flow_pan':f(t),
        'flow_tilt':f(t),'flow_roll':f(t),'gyro_gx':-f(t-.3)})
    mapping={name:{'axis':'gyro_gx','sign':-1} for name in ['flow_pan','flow_roll']}
    traces=aligned_motion(table,{'axis_mapping':mapping,'offset_s':.3})
    for grid,video,gyro in traces.values():
        valid=np.isfinite(video)&np.isfinite(gyro)&(grid>1)&(grid<18)
        assert np.corrcoef(video[valid],gyro[valid])[0,1]>.999
    assert aligned_motion(table,{})=={}
