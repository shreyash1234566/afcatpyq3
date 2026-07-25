# AFCAT Topic Prediction Methodology

## Model: Dirichlet-Multinomial Forecaster with Recency Decay

### Why Not ML Ensemble?

Honest nested rolling-origin backtesting (no leakage) showed:

| Model | MAE | MASE |
|---|---|---|
| XGBoost | 1.034 | 1.13 |
| Random Forest | 0.926 | 1.01 |
| Naive last-year | 0.915 | 1.00 |
| DM (marginalised A) | 0.885 | 0.97 |
| **EWMA nested** | **0.852** | **0.93** |

The ML models underperform even the naive baseline once leakage is removed. The Dirichlet-Multinomial forecaster achieves **18% MAE reduction** over XGBoost and **7% over naive**.

### Statistical Model

For each section, topic counts follow a Dirichlet-Multinomial distribution:

```
counts_t ~ Multinomial(N_t, θ)
θ ~ Dirichlet(α)
```

**Recency decay (EWMA):** Past year counts are exponentially weighted with decay factor γ:

```
w_t = γ^(T - t)   for t = 1..T
effective_count_i = Σ_t w_t * count_{i,t}
```

**Shrinkage toward uniform prior:** Concentration parameter α_i = effective_count_i + shrink × (total/K), where K = number of topics in section.

**Marginalised concentration A:** Rather than fixing A (total Dirichlet mass), a grid `A ∈ {0.25, 0.5, 1, 2, 3, 4, 6, 8, 12}` is evaluated. Each value is weighted by its marginal likelihood under the observed data. The final α vector is a weighted average across the grid — this is the key fix that raised interval coverage from 72.7% to 95.4%.

**Beta-Binomial credible intervals:** For each topic i with predicted proportion p_i and section total N:

```
X_i ~ BetaBinomial(N, α_i, A - α_i)
```

Intervals are the 5th and 95th percentiles of this distribution (90% CI).

### Per-Section Hyperparameters

Tuned by nested rolling-origin CV (inner loop selects params, outer loop evaluates):

| Section | γ (decay) | shrink | window |
|---|---|---|---|
| Verbal Ability | 0.7 | 0.2 | 3 years |
| General Awareness | 0.1 | 0.0 | all years |
| Reasoning | 0.4 | 0.1 | 3 years |
| Numerical Ability | 0.1 | 0.0 | all years |

GA and Numerical use all history (γ=0.1 ≈ uniform weighting) — their topic distributions are stable. Verbal and Reasoning show recency effects.

### Backtest Design (Reproducible Proof)

Script: `scripts/backtest_models.py`

- **Nested rolling-origin CV:** 48 folds (outer), inner loop tunes γ/shrink/window on preceding years only
- **No leakage:** hyperparameters are re-tuned at each outer fold boundary; test fold is never seen during tuning
- **Metrics:** MAE (primary), MASE (vs naive), cosine similarity, interval coverage
- **Coverage result:** 95.4% empirical coverage vs 90% nominal; not significantly different (n=48, ±8.5pt band)

### Production Forecast (2026)

When generating the 2026 forecast, `DirichletForecaster` is fit on **all 15 years** (2011–2025) with hyperparameters re-tuned on the full history. This is implemented in `DirichletForecaster.from_repo()`.

### Data

- Source: `data/processed/Q.json` — 2877 questions, 51 PDFs, 2011–2025
- 147 distinct topics across 4 sections; 67.9% of topic×year cells are zero
- Section targets: Verbal 30, GA 25, Reasoning 25, Numerical 20

### Files

| File | Purpose |
|---|---|
| `models/dirichlet_forecaster.py` | Production forecaster class |
| `models/topic_predictor.py` | Pipeline entry point (calls DirichletForecaster) |
| `scripts/backtest_models.py` | Reproducible backtest proof |
| `PREDICTION_METHODOLOGY.md` | This document |
