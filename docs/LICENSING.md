# Licence gate

Policy is data-driven in contracts/licenses.yaml. Eligibility is independent from task quality and gyro verdicts. Intended use defaults to research; each card records permissions, credit, eligibility, and public exclusions. An unknown licence allows analysis only.

The NEW data/manifest_v2.csv merges the old manifests while preserving them unchanged. Source is true provenance; collection preserves the old grouping. Existing own-labelled downloaded samples are egodex or epic. media_modified is determined by parent-video byte differences, not task labels. No genuinely self-recorded footage is currently in the catalogue.

No-modification licences never create export derivatives. Original-only sources use original bytes, and modified variants are excluded from public reports, app demo sets and submission packages. Privacy blur that conflicts with a no-modification licence withholds the media. Proposed cuts remain shaded on an original. Media credits are shown in the report. Retired public assets are moved to outputs/retired_public/ rather than destroyed; that directory is not served by the app.

Audit: outputs/licenses/eligibility.csv lists every catalogue row. The rebuilt home demo archive follows the policy; original-only videos retain original bytes instead of 480p transcoding. Tests cover the permissions matrix and ensure that quality verdicts do not change.

Own footage only: `uv run python tools/make_variants_from_own.py /path/to/my_clip.mp4 --name my_clip --split train`. This command declares self-recorded provenance, writes variants under outputs/own_variants/, and registers own rows in manifest_v2. It refuses a catalogue ID belonging to another source. Insertions do not fabricate or blindly copy sensor data.

The licence YAML reflects the requested application policy, not independent legal advice or a grant of rights. Original source terms and truthful provenance remain the operator's responsibility.
