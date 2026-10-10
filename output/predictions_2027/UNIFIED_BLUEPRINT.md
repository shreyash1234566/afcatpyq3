# Unified January 2027 blueprint

`scripts/build_unified_2027.py` fits history excluding the latest August anchor,
blends August 2026 topic shares, and calls `blueprint()` for integer macro/micro
allocation. The generator entry point delegates to this pipeline. Dashboard
compilation calls the same blueprint and fails on any mock count mismatch.

The current paper has 94 positive micro-topic slots totaling 100 questions:
Verbal 30, General Awareness 25, Reasoning 25, Numerical 20. Question labels are
matched exactly; original passages and figures are preserved. Practice selection
uses only original PYQs, randomized within the newest eligible source year with
a reproducible seed. Missing answer keys for selected older questions receive
separate labelled worked solutions, without modifying the source question bank.
No AI-generated stems or numeric mutations are used. This is
a blueprint-aligned practice paper, not a claim to predict exact exam text.

At the user's request, 2026 is excluded from transition-weight calibration but
August 2026 is still the latest observed anchor. Average pre-2026 per-transition
macro-error-optimal August weights: Verbal 46.67%, GA 52.50%, Reasoning 66.67%,
Numerical 51.67%; remainder is historical model. This arithmetic mean of optima
is not necessarily the aggregate-error-minimizing weight or micro-coverage optimum.

The review extends back to 2016, but year-only 2017–2021 files cannot establish
cycle transitions. Usable sensitivity pairs are currently into 2023–2025, with
source-label/block-recovery caveats; GA has only two pairs. Excluding 2026 is a
scenario choice, not statistical proof it is anomalous. No new held-out accuracy
is claimed. Prior PDF and DM scores do not validate this new weighted forecast.

Run `python scripts/audit_cross_year_sensitivity.py`, then
`python scripts/generate_2027_sota.py`, then
`python -m unittest discover -s tests -p test_shift_validation.py`.
Source cycle-date verification and independent future validation remain gaps.
