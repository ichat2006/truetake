# Final submission audit

Step 1: `outputs/final/agreement.md` compares only TEST rows in the root manifest.
All 11 expected completion labels are unknown (0 verified, 11 unverified); six
rows have no pipeline result. Two available clips need review. There is no
comparable labeled denominator, so no agreement percentage is claimed. No rule
or threshold was changed. `docs/final_agreement.py` reproduces the audit.

Step 2: public/internal reports now have five thumbnail tiles and a verdict/sensor legend. The actual stops are black_gap (pending trim), splice (rejected), controlled duplicate, P03_119 (hands close call), and P01 wrong_task (unconfirmed). No approved trimmed source or confirmed rejected wrong-task source exists; both gaps are labeled on the page. No policy changes or new AI calls.

Step 3: concise judges README replaces outdated calibration/deferred claims. Credits distinguish downloaded EgoDex footage from actual own recordings. The unchanged calibrated EPIC profile is copied into docs/ for reproducible code-only setup. Timing is labeled an estimate; model escalation and evaluation gaps are explicit.

Step 4: all requested ignore cases pass, including standalone Rerun files. Tracked-path audit: no matches; history credential-prefix count: 0. No history was rewritten. Detailed sanitized results: outputs/final/safety.json.
