# Final submission audit

Step 1: `outputs/final/agreement.md` compares only TEST rows in the root manifest.
All 11 expected completion labels are unknown (0 verified, 11 unverified); six
rows have no pipeline result. Two available clips need review. There is no
comparable labeled denominator, so no agreement percentage is claimed. No rule
or threshold was changed. `docs/final_agreement.py` reproduces the audit.

Step 2: public/internal reports now have five thumbnail tiles and a verdict/sensor legend. The actual stops are black_gap (pending trim), splice (rejected), controlled duplicate, P06_108 (gyro close call), and P01 wrong_task (unconfirmed). No approved trimmed source or confirmed rejected wrong-task source exists; both gaps are labeled on the page. No policy changes or new AI calls.
