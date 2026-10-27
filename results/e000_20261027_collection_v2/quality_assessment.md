## Repeatability and timing audit

Prepared-input mismatches: 0/120 cells. Parsed-answer disagreements: 8/120 cells.

Flagged question/budget cells: 197-3 at Uniform-8, 002-3 at Uniform-8, 098-1 at Uniform-8, 034-1 at Uniform-48, 235-1 at Uniform-8, 235-1 at Uniform-32, 235-3 at Uniform-8, 235-3 at Uniform-48.

Budgets exceeding the proposed median relative timing range of 20%: [8, 16, 32, 48, 64]. Individual timing cells above 50% remain flagged in `summary.json`; completed trials were never selectively replaced.

Uniform-64 observed PyTorch peaks: 10.39 GiB allocated, 16.73 GiB reserved. Sampled device usage reached 15.95 GiB. Reserved memory is allocator accounting; device sampling is a separate observation. These measurements do not identify the cause of the timing or answer disagreements.

## Descriptive task-quality assessment

120/120 repetition-1 video outputs met the fixed answer format; text-only: 23/24.

Video accuracy ranged from 58.3% to 87.5% across the five budgets; text-only scored 14/24 (58.3%). These are descriptive comparisons on 24 questions clustered in eight videos. Repetitions do not increase sample size. Useful visual reasoning, adaptive sampling, and statistical generalization are not established.

Questions incorrect under every video budget: 002-2, 034-2, 235-1. Every question is listed below; errors are counted over the five conditions, without selecting repetitions.

| Source | Question | Incorrect video conditions / 5 | Invalid conditions / 5 | Text-only correct |
|---|---|---:|---:|---|
| 1q-5IIyZL20 | 197-1 | 0 | 0 | False |
| 1q-5IIyZL20 | 197-2 | 0 | 0 | True |
| 1q-5IIyZL20 | 197-3 | 2 | 0 | True |
| N1cdUjctpG8 | 002-1 | 1 | 0 | False |
| N1cdUjctpG8 | 002-2 | 5 | 0 | False |
| N1cdUjctpG8 | 002-3 | 1 | 0 | True |
| PU-XOFIJMlg | 098-1 | 1 | 0 | True |
| PU-XOFIJMlg | 098-2 | 2 | 0 | True |
| PU-XOFIJMlg | 098-3 | 0 | 0 | False |
| SnOc5W0PgVE | 078-1 | 0 | 0 | True |
| SnOc5W0PgVE | 078-2 | 0 | 0 | True |
| SnOc5W0PgVE | 078-3 | 0 | 0 | True |
| WViSvPFUVd8 | 079-1 | 0 | 0 | True |
| WViSvPFUVd8 | 079-2 | 0 | 0 | True |
| WViSvPFUVd8 | 079-3 | 0 | 0 | True |
| blSnLEZe-sI | 034-1 | 3 | 0 | True |
| blSnLEZe-sI | 034-2 | 5 | 0 | False |
| blSnLEZe-sI | 034-3 | 2 | 0 | False |
| kij9GXWdmik | 235-1 | 5 | 0 | False |
| kij9GXWdmik | 235-2 | 0 | 0 | True |
| kij9GXWdmik | 235-3 | 3 | 0 | True |
| v0m3E_uSyFg | 234-1 | 0 | 0 | False |
| v0m3E_uSyFg | 234-2 | 0 | 0 | False |
| v0m3E_uSyFg | 234-3 | 0 | 0 | False |

Reproduce this addendum with `python scripts/report_quality.py --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2`. Its code hash is recorded in `quality_assessment.json`.
