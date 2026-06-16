# Thesis Claim Boundary Audit

This audit scans thesis-facing drafts for unsupported overclaims. Guarded mentions inside boundary or do-not-say contexts are allowed.

- Status: `claim_boundaries_fail`
- Violations: `1`
- Guarded mentions: `0`
- Missing required phrases: `0`

## Required Route Phrases

| Phrase | Present |
| --- | --- |
| 已感知路面不平度 | True |
| PreScan | True |
| BP-NMPS | True |
| SI-RPPO | True |
| SAC | True |

## Violations

- `bad.md:2` `si_rppo_universal_superiority`: SI-RPPO is universally superior to SAC.

## Guarded Mentions

