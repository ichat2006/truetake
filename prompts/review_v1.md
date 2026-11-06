Review these first-person task images in timestamp order against the supplied contract.
Judge only visible evidence. Use unsure when the images do not establish the answer.
Write evidence observations before the verdict. Do not infer continuous visibility
or exact transition times between images. The garment must end folded into a compact
rectangle for goal=yes; an already folded initial garment alone does not prove folding.
Return only JSON with this structure:
{"evidence":[{"id":"e1","frame_id":0,"t":0.0,"observation":"visible fact"}],
 "goal":{"verdict":"yes|no|unsure","evidence_ids":["e1"],"reason":"brief reason"},
 "process":[{"rule":"hands_visible","verdict":"yes|no|unsure","evidence_ids":["e1"],"reason":"brief reason"}],
 "steps":[{"name":"smooth","evidence_ids":["e1"]}],
 "tags":{"garment":"tshirt|cloth|towel|other|unsure","surface":"table|bed|lap|other|unsure","leading_hand":"left|right|both|unsure"}}
The verdict field must be exactly yes, no, or unsure (not the pipe-separated example).
Allowed step names: smooth, sleeves-in, fold-half-long, fold-half-short, fold-thirds,
roll, flip, drop, recover, other. Cite actual supplied frame IDs and timestamps.
Definite goal verdicts and every step require evidence IDs. Empty steps are allowed.
