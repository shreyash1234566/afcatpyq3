# Micro-topic coverage audit

This is an expanding-window, pre-year test of the current code. It is not a reproduction of the supplied PDF audits.
Every predicted slot has an integer quota. Future years are excluded from evidence and parameter tuning.

| Year | Hit precision | Questions covered | Coverage | Predicted slots |
|---|---:|---:|---:|---:|
| 2014 | 32.8% | 80/200 | 40.0% | 61 |
| 2015 | 38.9% | 133/200 | 66.5% | 72 |
| 2016 | 39.4% | 60/100 | 60.0% | 66 |
| 2017 | 37.9% | 64/100 | 64.0% | 66 |
| 2018 | 35.4% | 54/100 | 54.0% | 65 |
| 2019 | 23.9% | 55/100 | 55.0% | 67 |
| 2020 | 30.8% | 44/100 | 44.0% | 65 |
| 2021 | 31.4% | 57/100 | 57.0% | 70 |
| 2022 | 51.5% | 150/315 | 47.6% | 68 |
| 2023 | 47.3% | 94/200 | 47.0% | 74 |
| 2024 | 72.2% | 277/485 | 57.1% | 79 |
| 2025 | 53.1% | 216/436 | 49.5% | 81 |
| 2026 | 33.7% | 51/300 | 17.0% | 83 |

| Subject | Hit precision | Questions covered | Coverage |
|---|---:|---:|---:|
| Verbal Ability | 35.9% | 383/781 | 49.0% |
| General Awareness | 28.7% | 153/594 | 25.8% |
| Reasoning | 64.6% | 580/831 | 69.8% |
| Numerical Ability | 45.0% | 219/530 | 41.3% |

## Definitions and limitations

- Hit precision: predicted nonzero micro-topic slots that appeared, divided by all predicted slots.
- Question coverage: observed questions whose section, normalized broad topic, and exact micro-topic label match a predicted slot. Reading these topics is not a guarantee of answering correctly.
- Recall: distinct observed slots that were predicted, divided by all distinct observed slots.
- Normalized-count MAE: absolute count error over the union of predicted and observed slots, after scaling each observed section to the target exam quota.
- Exact raw-count matches are included in JSON, but are not single-exam exact-count accuracy for years containing multiple papers.
- Missing/new labels count as misses; no test-year micro-topic vocabulary is used in predictions.
- Yearly totals pool papers, shifts, and partial memory-based records. They cannot be presented as coverage of one complete 100-question exam.
- Older weights and topic labels are retrospective choices. This is a code backtest, not a prospective independent evaluation.
- The supplied PDF audits use a particular frozen 2026 blueprint. These results use pre-year refits of the current algorithm; the two protocols and taxonomies need reconciliation before comparing accuracy.
- A macro allocation is fitted solely to obtain pre-year quotas; the production macro model and 2027 plan were not changed.
- No nominal calibration confidence band is claimed: folds and topics are dependent.
