# Thesis Claim Boundary Audit

This audit scans thesis-facing drafts for unsupported overclaims. Guarded mentions inside boundary or do-not-say contexts are allowed.

- Status: `claim_boundaries_pass`
- Violations: `0`
- Guarded mentions: `31`
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

- None.

## Guarded Mentions

- `docs\thesis_chapter_draft_cn.tex:6` `prescan_core_route`
- `docs\thesis_chapter_draft_cn.tex:32` `bp_nmps_replaces_mpc`
- `docs\thesis_chapter_draft_cn.tex:61` `si_rppo_universal_superiority`
- `docs\thesis_chapter_draft_cn.tex:78` `vla_completed_platform`
- `docs\thesis_chapter_draft_cn.tex:78` `bp_nmps_replaces_mpc`
- `docs\thesis_chapter_draft_cn.tex:78` `si_rppo_universal_superiority`
- `docs\thesis_chapter_draft_cn.md:7` `prescan_core_route`
- `docs\thesis_chapter_draft_cn.md:46` `bp_nmps_replaces_mpc`
- `docs\thesis_chapter_draft_cn.md:78` `si_rppo_universal_superiority`
- `docs\thesis_chapter_draft_cn.md:98` `vla_completed_platform`
- `docs\thesis_chapter_draft_cn.md:102` `bp_nmps_replaces_mpc`
- `docs\thesis_chapter_draft_cn.md:103` `si_rppo_universal_superiority`
- `docs\thesis_defense_qa.md:7` `prescan_core_route`
- `docs\thesis_defense_qa.md:9` `prescan_core_route`
- `docs\thesis_defense_qa.md:21` `bp_nmps_replaces_mpc`
- `docs\thesis_defense_qa.md:49` `si_rppo_universal_superiority`
- `docs\thesis_defense_qa.md:69` `vla_completed_platform`
- `docs\thesis_defense_qa.md:71` `vla_completed_platform`
- `docs\thesis_defense_qa.md:73` `chrono_vla_platform`
- `docs\thesis_defense_qa.md:77` `si_rppo_universal_superiority`
- `docs\si_rppo_ccfa_innovation_framing.md:57` `si_rppo_universal_superiority`
- `docs\si_rppo_ccfa_innovation_framing.md:68` `si_rppo_universal_superiority`
- `docs\si_rppo_ccfa_innovation_framing.md:70` `vla_completed_platform`
- `docs\si_rppo_ccfa_innovation_framing.md:79` `si_rppo_universal_superiority`
- `docs\thesis_final_submission_checklist.md:42` `prescan_core_route`
- `docs\thesis_final_submission_checklist.md:43` `chrono_vla_platform`
- `docs\thesis_final_submission_checklist.md:43` `vla_completed_platform`
- `docs\thesis_final_submission_checklist.md:45` `bp_nmps_replaces_mpc`
- `docs\thesis_final_submission_checklist.md:47` `si_rppo_universal_superiority`
- `docs\thesis_final_submission_checklist.md:60` `chrono_vla_platform`
- `docs\thesis_final_submission_checklist.md:60` `vla_completed_platform`
