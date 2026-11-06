# Gyroflow portable trial

Gyroflow v1.6.3 was downloaded from the official GitHub release and extracted
under `outputs/tools/gyroflow-v1.6.3/`. No system packages or Python dependencies
were installed. Missing Ubuntu libc++, libc++abi, and libunwind runtime packages
were downloaded and extracted into the portable tool's `runtime/` directory.

Release asset: https://github.com/gyroflow/gyroflow/releases/download/v1.6.3/Gyroflow-linux64.tar.gz

Verified SHA-256: `98625e3ce318964d49d3c66f2dc302a9e6c4c138233d9aa6cbd986f687c421c3`.

## What ran

- `--help` ran successfully after supplying the local runtime libraries.
- EPIC video plus GCSV import/export completed in 0.526 seconds.
- An initial startup stalled during GPU discovery and was stopped by timeout.
- A longer synchronization attempt also timed out.
- A reduced run completed in 4.891 seconds: three evenly spaced sync points,
  1.5 seconds per point, every second video frame, 360 processing-resolution
  setting, and a 2-second search range. Two offsets were exported; a third
  candidate was rejected as outside the acceptable range.

Artifacts:

- `outputs/gyroflow_comparison/epic_import.gyroflow`
- `outputs/gyroflow_comparison/epic_sync.gyroflow`
- `outputs/gyroflow_comparison/epic_import.log`
- `outputs/gyroflow_comparison/epic_sync.log`

Gyroflow reported offsets of −231.01 ms and −1263.01 ms near the 10-second
and 30-second sync points. These are Gyroflow's own offset convention; do not
assume its sign convention equals the custom estimator's. The disagreement,
unconfirmed IMU orientation, and missing lens calibration mean these results
are exploratory and must not be used as a verification pass or calibration.

The GCSV preserves the source axes with an explicitly unconfirmed `XYZ`
orientation. Timestamp and gyro scales are 1; acceleration scale is
1/9.80665 because Gyroflow expects g rather than m/s². Imported source is
identified in the project as `gcsv truetake epic P01 102`.

## Dataset compatibility finding

The [official EPIC extension acquisition guide](https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/AcquisitionGuidelines/EPIC-Kitchens-acquisition-guidelines.pdf)
specifies GoPro Hero 7, 1080p/50fps, Linear field of view, and **video
stabilization ON**. [Gyroflow's camera documentation](https://github.com/gyroflow/docs.gyroflow.xyz/blob/main/getting-started/supported-cameras/gopro.md)
requires internal stabilization OFF for GoPros before Hero 8.

That is a material protocol mismatch. The specific clip's stabilization setting
has not been independently confirmed, but it must not be assumed to satisfy
Gyroflow's supported input requirements. Our re-encoded excerpt also lacks
the original embedded telemetry and lens metadata; separate gyro/acceleration
CSVs were supplied. The exported project contains an empty lens calibration,
not a confirmed camera profile.

Do not replace TrueTake's verifier based on this trial. First validate an
unstabilized recording with known lens and IMU orientation using Gyroflow.
Retain original sensor measurements when evaluating consistency; optical
correction that changes them would compromise their role as independent evidence.

## Resource controls and status

No Gyroflow process remained after the trials. At the subsequent check WSL
reported approximately 6.8 GiB available memory and no swap usage. These
observations do not establish what caused the application crash the user saw.

The local `run.sh` now disables OpenCL/WGPU discovery, limits the process to
two allowed CPUs and 30 seconds of CPU time, applies a 60-second wall timeout
with a five-second termination grace period, and disables core dumps. Tool
settings/cache/data use its own output directory. The successful reduced trial
preceded the final hard resource guards; no new synchronization run was started
after the user reported a crash. The reduced trial still logged failed hardware
decoder probes before proceeding despite `--no-gpu-decoding`.

Only lightweight artifact and shell-syntax validation followed the crash report.
The existing TrueTake test suite last passed 98 tests; its production source was
not changed by this portable-tool trial.
