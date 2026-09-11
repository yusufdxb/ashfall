# Historical reset-file curriculum

The n=11 experiment used failure-associated reset files. Phoenix's default reset loaded the first trajectory row, while the synthetic trajectories contained stable pre-failure frames. Root velocity and command restoration were not enabled. The experiment therefore does not establish that failure-onset-conditioned repair is ineffective.

The original aggregate metrics and reports are preserved byte for byte in [results/legacy_row0_curriculum](../../results/legacy_row0_curriculum). The provenance index maps original paths to archived paths and SHA-256 hashes. Interpret original reports as historical documents: their claims are superseded by the [implementation audit](../audits/legacy_ashfall_audit.md). Historical scripts accept this archive directory as their results input. Existing `failure_fraction` names in archived configs identify the old treatment and retain their original meaning.

The 18 synthetic positives cover detector thresholds and file integration. They do not independently validate detection on physical failures. No hardware-counterexample repair result is established by this archive. Non-significant treatment differences do not demonstrate equivalence.
