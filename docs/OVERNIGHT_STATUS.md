# Overnight implementation status

Follow-up: the active folding/washing contracts now measure at least one detected
hand (`hand_present`, target 0.60, review band 0.10). `hands_inside` remains unchanged.
Decisions and exports can be rebuilt without inference or API calls using
`uv run python tools/recompute_cached_demo.py --rerun`. Full before/after audits are
under `outputs/hand_presence/`. The hands rule keeps its historical ID for cached
evidence references; its description and measurement define the current policy.
Locally confirmed unusable intervals may be cut in a pending review copy, which
remains excluded from the accepted training collection. Idle and goal rules are
unchanged. The counts below describe the earlier overnight policy.

The detailed local audit is `outputs/overnight/REPORT.md` (outputs are not committed).

The requested Haiku 5.5 and Sonnet 5.5 probes could not pass SDK authentication
validation. Credential files were not opened. Existing cached reviews are preserved.
Model availability has not been established. Validation retry, escalation and a new
paid corpus review remain outstanding; no model-upgrade code was retained.

The strict measured-target policy rejects the current independent corpus. This is
reported as observed, with no targets tuned on test clips and no fabricated approvals.

Final validation: 235 tests passed, two real-clip CLI checks passed, three Rerun
recordings validated, and headless Chrome played the new black-gap original in the
rebuilt report. Public/internal reports and the public backup are under `outputs/`.
There are 26 evaluated rows: 25 rejected and one controlled duplicate. No approved
cut example, approved EPIC card, or final NEEDS REVIEW card exists under this policy.
