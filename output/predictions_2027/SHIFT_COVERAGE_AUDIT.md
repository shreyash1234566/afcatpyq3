# Chronological paper-by-paper validation

| Paper | Exact-label coverage | Canonical macro lookup | Study-family coverage |
|---|---:|---:|---:|
| AFCAT 2014 Official Paper1.pdf | 46/100 | 46/100 | 47/100 |
| AFCAT 2014 Official Paper2.pdf | 34/100 | 34/100 | 38/100 |
| AFCAT 2015 Official Paper1.pdf | 65/100 | 65/100 | 73/100 |
| AFCAT 2015 Official Paper2.pdf | 67/100 | 67/100 | 73/100 |
| AFCAT 2016 Official Paper2.pdf | 60/100 | 62/100 | 66/100 |
| AFCAT 2017 Memory.pdf | 64/100 | 65/100 | 73/100 |
| AFCAT 2018 Memory.pdf | 54/100 | 56/100 | 60/100 |
| AFCAT 2019 Memory.pdf | 55/100 | 56/100 | 69/100 |
| AFCAT 2020 Memory.pdf | 44/100 | 49/100 | 53/100 |
| AFCAT 2021 Memory.pdf | 57/100 | 59/100 | 68/100 |
| AFCAT 2023 01 Memory.pdf | 48/100 | 49/100 | 59/100 |
| AFCAT 2023 02 Memory.pdf | 44/100 | 45/100 | 48/100 |
| AFCAT 2026 Jan 31 Shift 1 | 46/100 | 46/100 | 49/100 |
| AFCAT 2026 Aug Shift 1 | 10/100 | 10/100 | 16/100 |
| AFCAT 2026 Aug Shift 2 | 9/100 | 9/100 | 15/100 |

## Aggregate metrics

| Method | Papers | Topic precision | Question coverage |
|---|---:|---:|---:|
| current_literal | 15 | 27.8% | 46.9% |
| canonical_macro_lookup | 15 | 28.1% | 47.9% |
| study_family_coverage | 15 | 31.2% | 53.8% |
| regime_2026_base | 2 | 8.6% | 9.5% |
| regime_2026_family | 2 | 12.5% | 15.5% |

## January-to-August 2026 comparison

| Paper | Without January | With January (canonical lookup) | 2026-base candidate |
|---|---:|---:|---:|
| AFCAT 2026 Aug Shift 1 | 10/100 | 10/100 | 10/100 |
| AFCAT 2026 Aug Shift 2 | 9/100 | 9/100 | 9/100 |

## Limitations

- Only papers with 100 unique numbered questions are scored.
- Same-day papers never train each other; unknown dates exclude overlapping intervals.
- Date of exam is not proof of date of availability; labeling was retrospective.
- Partial valid earlier papers can enter training; invalid merged papers are excluded.
- Study-family scores are coarser and cannot replace exact micro-topic scores.
- Historical subject counts differ from current fixed targets; count metrics are most comparable for 2026.
- The 2026-base candidate has only two out-of-sample August shifts, not enough to establish January 2027 performance.