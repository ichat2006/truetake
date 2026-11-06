# Final submission audit

Step 1: `outputs/final/agreement.md` compares only TEST rows in the root manifest.
All 11 expected completion labels are unknown (0 verified, 11 unverified); six
rows have no pipeline result. Two available clips need review. There is no
comparable labeled denominator, so no agreement percentage is claimed. No rule
or threshold was changed. `docs/final_agreement.py` reproduces the audit.

Step 2: public/internal reports now have five thumbnail tiles and a verdict/sensor legend. The actual stops are black_gap (pending trim), splice (rejected), controlled duplicate, P03_119 (hands close call), and P01 wrong_task (unconfirmed). No approved trimmed source or confirmed rejected wrong-task source exists; both gaps are labeled on the page. No policy changes or new AI calls.

Step 3: concise judges README replaces outdated calibration/deferred claims. Credits distinguish downloaded EgoDex footage from actual own recordings. The unchanged calibrated EPIC profile is copied into docs/ for reproducible code-only setup. Timing is labeled an estimate; model escalation and evaluation gaps are explicit.

Step 4: all requested ignore cases pass, including standalone Rerun files. Tracked-path audit: no matches; history credential-prefix count: 0. No history was rewritten. Detailed sanitized results: outputs/final/safety.json.

Step 5: offline submission folder contains 26 thumbnail/reason cards, but video only for five Start-here clips (8 MP4s including three existing derivatives). All media are H.264/yuv420p, 480p, CRF 30, faststart and silent. All local asset links and card anchors validate; restricted-source identifier scan is empty. The agreement copy omits restricted identifiers while preserving aggregate counts.

Demo zip: 28,544,970 bytes (28.54 MB), below 100 MB. Chrome file-URL playback passed for the 854×480 trimmed copy, duration 22.07s. Two documented real-clip CLI checks passed. Full suites by step: 245 / 246 / 246 / 246 / 247 passed. No API calls, detector reruns, dependency additions, data edits or history rewrites.

Start-here IDs: basic_fold_1_black_gap; splice_P03_109_P04_104; basic_fold_0_controlled_reupload (compare basic_fold_0); P03_119_207_267; P01_102_000_060_wrong_task.

Remaining: unknown completion labels prevent agreement measurement; no approved trimmed example; wrong-task goal remains unverified. README screenshot/event-history placeholders remain for the author. EgoDex footage is not author-recorded footage; source NC-ND terms remain applicable to adapted demo assets and should be checked before public redistribution. Current package metadata and archive sizes are recorded under outputs/final/.
