# Proposals and historical reports

This index preserves the supplied source files in their original locations.

## Current authority

- [Revised execution plan v2](../Weather_Robust_ADAS_Execution_Plan_v2.pdf) —
  focused 13-week plan, 17 August–15 November 2026.
- [Original TiHAN proposal](../sem5-works/Proposal%201%20-%20TiHAN%20IITH%20_.pdf) —
  broad eight-month parent project.

## Semester-5 team work

- [Research progress report](../sem5-works/report_sem5_research_.pdf)
- [Code report](../sem5-works/research_code_report_1_.pdf)
- [Proposed workflow](../sem5-works/proposed_workflow_.png)
- [Historical Colab notebook](../baseline.ipynb)

The team report's technical contribution is attention gating on U-Net skip
connections in a Pix2Pix smoke-restoration generator. See
[PROJECT_AUDIT.md](PROJECT_AUDIT.md) before quoting its metrics.

## Internship work

- [Initial report 1](../intern-work/init_report1_.docx)
- [Initial report 2](../intern-work/init_report_2_.pdf)
- [July 2025 monthly report](../intern-work/Monthly%20Progress%20Report-July-2025_.pdf)

These reports contain useful dataset and Colab links and claims for rain, snow,
fog, smoke, detection, and Raspberry Pi deployment. Because their runnable code,
checkpoints, split lists, and raw logs are absent, all numerical claims are
labelled “historical/unverified” until reproduced.

## Document control

New plans, meeting decisions, and experiment reports should be Markdown in
`docs/`. Do not edit the supplied PDFs. A result becomes “verified” only when its
run directory contains the config, Git commit, split manifest hash, checkpoint,
environment, and raw metric output.
