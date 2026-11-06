"""Retimestamp cached evidence for a documented black insertion, never other edits."""
from copy import deepcopy
import numpy as np
import pandas as pd

INSERT_AT_S = 8.0
INSERT_DURATION_S = 5.0


def shifted_time(t: float) -> float:
    return float(t + INSERT_DURATION_S if t >= INSERT_AT_S else t)


def transfer_insert_cache(parent: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Reuse unchanged content; inserted rows have no inherited semantic evidence or IMU."""
    result=target.copy()
    gap=(result.t>=INSERT_AT_S)&(result.t<INSERT_AT_S+INSERT_DURATION_S)
    source_times=np.where(result.t>=INSERT_AT_S+INSERT_DURATION_S,result.t-INSERT_DURATION_S,result.t)
    indices=np.abs(parent.t.to_numpy()[:,None]-source_times[None,:]).argmin(axis=0)
    protected=set(result.columns)
    for column in parent.columns:
        if column in protected or column in {'gyro_alignment','claude_review','claude_usage'}:continue
        values=parent[column].iloc[indices].copy().reset_index(drop=True)
        result[column]=values
        if column not in {'detector_stats'}:
            if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
                result.loc[gap,column]=np.nan
            else:
                result[column]=result[column].astype(object)
                result.loc[gap,column]=None
    result['sensor_present']=False
    result['imu_valid']=False
    for column in ['flow_pan','flow_roll','flow_tilt','flow_t','flow_dt']:
        if column in result:
            result.loc[gap,column]=np.nan
    if 'flow_t' in result:result.loc[~gap,'flow_t']=result.loc[~gap,'flow_t'].map(shifted_time)
    if 'people_status' in result:result.loc[gap,'people_status']='skipped_dark'
    if 'flow_status' in result:result.loc[gap,'flow_status']='gap'
    if 'thumbnail_path' in result:result['thumbnail_path']=None
    review=deepcopy(parent.iloc[0].get('claude_review'))
    if review:
        id_map={}
        for evidence in review['evidence']:
            t=shifted_time(evidence['t'])
            row=int(np.abs(result.t.to_numpy()-t).argmin())
            id_map[evidence['frame_id']]=row
            evidence.update(t=float(result.at[row,'t']),frame_id=row)
        old_rules={r['id']:r for r in parent.iloc[0]['contract']['process']}
        for rule in review.get('rules',[]):
            if old_rules.get(rule['rule_id'],{}).get('measurement')=='hand_present':rule['rule_id']='one_hand_visible'
        for interval in review.get('intervals',[])+review.get('follow_up_requests',[]):
            interval['start_s']=shifted_time(interval['start_s'])
            interval['end_s']=shifted_time(interval['end_s'])
        provenance=review.setdefault('provenance',{})
        provenance['source_review_reused_for_known_insertion']=True
        for key in ['followup_frame_ids','initial_frame_ids']:
            if key in provenance:provenance[key]=[id_map[i] for i in provenance[key] if i in id_map]
        from truetake.detect_claude import validate_contract_review
        frames={int(e['frame_id']):float(e['t']) for e in review['evidence']}
        validate_contract_review(review,frames,result.iloc[0]['contract'],float(result.sample_end.iloc[-1]))
    for column,value in [('claude_review',review),('claude_status','cached_known_insertion'),
                         ('claude_usage',{'actual_api_calls':0,'actual_input_tokens':0,'actual_output_tokens':0,
                                          'review_source':'parent cache, timestamps shifted for known insertion'})]:
        result[column]=pd.Series([value]+[None]*(len(result)-1),dtype=object)
    return result
