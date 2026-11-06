"""Transparent human-review estimates; cached usage is not charged a second time."""
import math

ASSUMED_MINUTES_PER_CLIP = 2.0
ASSUMED_REVIEWER_USD_PER_HOUR = 25.0
PROJECTION_CLIPS = 1000


def estimate(total: int, counts: dict, minutes_per_clip: float=ASSUMED_MINUTES_PER_CLIP,
             reviewer_rate: float=ASSUMED_REVIEWER_USD_PER_HOUR, usage: dict | None=None) -> dict:
    if not all(math.isfinite(v) and v>=0 for v in [minutes_per_clip,reviewer_rate]):
        raise ValueError('Review time and hourly rate must be finite, nonnegative numbers')
    automatic=sum(counts.get(key,0) for key in ['APPROVED','CUT','REJECTED'])
    duplicates=counts.get('DUPLICATE',0)
    human=max(0,total-automatic-duplicates)
    without=total*minutes_per_clip/60
    with_tt=human*minutes_per_clip/60
    saved=without-with_tt
    usage=usage or {'calls':0,'input_tokens':0,'output_tokens':0,'cost_usd':0.}
    cost=usage.get('cost_usd');labor=saved*reviewer_rate
    scale=PROJECTION_CLIPS/total if total else 0.
    return {'total_clips':total,'automatic':automatic,'human':human,'duplicates':duplicates,
            'minutes_per_clip':minutes_per_clip,'reviewer_rate':reviewer_rate,
            'hours_without':without,'hours_with':with_tt,'hours_saved':saved,
            'percent_saved':100*saved/without if without else 0.,'labor_saved_usd':labor,
            'ai_cost_usd':cost,'net_saved_usd':labor-cost if cost is not None else None,
            'usage':usage,'projection_clips':PROJECTION_CLIPS,
            'projection':{'hours_without':without*scale,'hours_with':with_tt*scale,
                          'labor_saved_usd':labor*scale,'ai_cost_usd':cost*scale if cost is not None else None}}
