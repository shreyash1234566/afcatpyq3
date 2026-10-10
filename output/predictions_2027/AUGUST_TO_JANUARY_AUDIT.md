# August -> next January/February validation

| Target | Previous-August questions | Regular exact coverage | August-anchor exact coverage | Regular family coverage | Anchor family coverage |
|---|---:|---:|---:|---:|---:|
| AFCAT 2026 Jan 31 Shift 1 | 207 | 46/100 | 40/100 | 49/100 | 46/100 |

## Observed macro changes for AFCAT 2026 Jan 31 Shift 1

| Subject | Topic | August share | January share | Change (percentage points) |
|---|---|---:|---:|---:|
| Verbal Ability | Para Jumbles | 25.0% | 10.0% | -15.0 |
| Verbal Ability | Fill in the Blanks | 0.0% | 13.3% | +13.3 |
| Verbal Ability | Reading Comprehension | 21.7% | 10.0% | -11.7 |
| Verbal Ability | One Word Substitution | 5.0% | 13.3% | +8.3 |
| Verbal Ability | Idioms & Phrases | 6.7% | 10.0% | +3.3 |
| General Awareness | Current Affairs | 6.5% | 24.0% | +17.5 |
| General Awareness | General Science | 13.0% | 4.0% | -9.0 |
| General Awareness | Environment/Ecology | 8.7% | 0.0% | -8.7 |
| General Awareness | Organizations | 6.5% | 0.0% | -6.5 |
| General Awareness | Defence | 19.6% | 24.0% | +4.4 |
| Reasoning | Logical Reasoning | 40.7% | 20.0% | -20.7 |
| Reasoning | Coding-Decoding | 6.8% | 24.0% | +17.2 |
| Reasoning | Number/Letter Series | 16.9% | 0.0% | -16.9 |
| Reasoning | Non-Verbal Pattern | 16.9% | 32.0% | +15.1 |
| Reasoning | Direction Sense | 0.0% | 8.0% | +8.0 |
| Numerical Ability | Statistics | 0.0% | 10.0% | +10.0 |
| Numerical Ability | Average | 9.5% | 0.0% | -9.5 |
| Numerical Ability | Ratio & Proportion | 14.3% | 5.0% | -9.3 |
| Numerical Ability | Mensuration | 7.1% | 15.0% | +7.9 |
| Numerical Ability | Time & Work | 2.4% | 10.0% | +7.6 |

## Limitations

Too few reliable transitions to learn or certify a reusable cycle-change rule. The 70/30 blend is a historical configuration, not an estimated optimum.
- Anchor sets can be partial memory-based papers.
- Exact micro-label scores and coarser study-family scores must remain separate.
- 2023 first-cycle filenames have no confirmed month; they are not silently treated as January.
- Multiple shifts from the same cycle are dependent, not independent training years.

## Exploratory weight sweep (NOT held-out winning-model accuracy)

| Subject | Best August weight for macro count error | Best August weight for micro coverage |
|---|---:|---:|
| Verbal Ability | 30% | 0% |
| General Awareness | 55% | 10% |
| Reasoning | 5% | 5% |
| Numerical Ability | 15% | 65% |

Choosing and reporting a winning weight on this one pair is tuning, not validation. Macro-count and micro-coverage objectives may choose different weights. No automatically learned fluctuation correction is deployed from one pair.

## Direction comparison: macro distribution change

Total variation = half the sum of absolute topic-share differences. This is distribution movement, not accuracy.
| Subject | Aug 2025 -> Jan 2026 | Jan 2026 -> Aug 2026 |
|---|---:|---:|
| Verbal Ability | 31.7% | 6.7% |
| General Awareness | 34.0% | 30.0% |
| Reasoning | 43.2% | 8.0% |
| Numerical Ability | 31.4% | 22.5% |