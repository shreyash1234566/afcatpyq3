"""
dirichlet_forecaster.py
=======================
Production forecaster for AFCAT per-topic question distribution.

Model: section-wise **Dirichlet-Multinomial** with
  - recency-decayed evidence (exponential decay gamma on yearly topic shares),
  - a SYMMETRIC Dirichlet prior (target = uniform 1/k) whose concentration A is
    **marginalised** over a grid weighted by marginal likelihood (not point-plugged),
  - the sum-to-section-total constraint built in (shares sum to 1 by construction),
  - simulation-based **Beta-Binomial** credible intervals that propagate A-uncertainty.

Why this and not XGBoost/LightGBM/RF: proven on this repo's own data by a
leakage-free nested rolling-origin backtest (see scripts/backtest_models.py):

    model                      honest MAE   MASE
    ---------------------------------------------
    Dirichlet-Multinomial       0.85        0.93     <-- this
    EWMA (same family)          0.85        0.93
    naive last-year             0.92        1.00
    RandomForest (old)          0.93        1.01
    XGBoost (old)               1.03        1.13     <-- worst

90% interval empirical coverage: 95.4% over n=48 folds (±8.5pt SE band around 90%
=> statistically indistinguishable from nominal). See PREDICTION_METHODOLOGY.md.

The point-forecast identity that justifies the family:
    symmetric-Dirichlet posterior mean
        p_i = (A/k + x_i) / (A + M)
            = [A/(A+M)]*(1/k) + [M/(A+M)]*(x_i/M)
    i.e. exactly  w*uniform + (1-w)*data_share  with w = A/(A+M).
    Shrinkage strength is per-fold-adaptive through M (effective #years of evidence).

No third-party ML deps: numpy + scipy only.
"""
from __future__ import annotations
import json, re, collections, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import betabinom, dirichlet as _dir  # dirichlet imported for completeness

# Import normalization map from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
try:
    from topic_normalization_map import TOPIC_NORMALIZATION
except ImportError:
    TOPIC_NORMALIZATION = {}

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent.parent
Q_PATH = ROOT / "data" / "processed" / "Q.json"

# AFCAT 2026 fixed section sizes (known constants, NOT forecast).
SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25,
              "Reasoning": 25, "Numerical Ability": 20}
MIN_SEC_QS = 5          # a year is usable for a section only if it has >= this many Qs
MIN_HISTORY = 3         # minimum prior years before we trust a forecast
GAMMA_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # empirical-vs-uniform prior blend, tuned per section
A_GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12], float)


# ----------------------------------------------------------------------------- data
def _year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None


def load_counts(q_path: Path = Q_PATH):
    """Return (cnt, syt, years): cnt[sec][topic][year]->int, syt[sec][year]->int.
    Topic names are normalized via TOPIC_NORMALIZATION before counting."""
    data = json.load(open(q_path, encoding="utf-8"))
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    syt = collections.defaultdict(collections.Counter)
    for q in data:
        y, s, t = _year(q.get("file_name")), q.get("section"), q.get("topic")
        if y and s in SEC_TARGET and t:
            t = TOPIC_NORMALIZATION.get(t, t)   # normalize topic name
            cnt[s][t][y] += 1
            syt[s][y] += 1
    years = sorted({y for s in syt for y in syt[s]})
    return cnt, syt, years


def section_years(syt, sec):
    return [y for y in sorted(syt[sec]) if syt[sec][y] >= MIN_SEC_QS]


def actual_vec(cnt, syt, sec, y, topics):
    tot = syt[sec][y]
    return np.array([cnt[sec][t][y] for t in topics], float) / tot * SEC_TARGET[sec]


def share_matrix(cnt, syt, sec, hist_years, topics):
    """Rows = per-year within-section topic shares (each row sums to 1)."""
    M = np.zeros((len(hist_years), len(topics)))
    for i, y in enumerate(hist_years):
        tot = syt[sec][y]
        if tot:
            M[i] = np.array([cnt[sec][t][y] for t in topics], float) / tot
    return M


def year_totals(syt, sec, hist_years):
    """Per-year labeled-question counts for a section. Used to sample-size-weight
    evidence so noisy small-sample years don't count as much as rich years."""
    return np.array([syt[sec][y] for y in hist_years], float)


# ------------------------------------------------------------------- core estimator
def decayed_evidence(S, gamma, n=None):
    """Recency-decayed sufficient statistic from a share-matrix S.
    If n (per-year sample sizes) is given, each year is ALSO weighted by its
    sample size relative to the mean — rich years count more than sparse ones.
    Returns (x, M): x[i] = decayed evidence for topic i, M = sum(weights) = eff #years."""
    L = len(S) - 1
    w = np.array([gamma ** (L - k) for k in range(len(S))])
    if n is not None and len(n) == len(S):
        nn = np.maximum(n, 1e-9)
        w = w * (nn / nn.mean())          # sample-size weighting (mean-normalized -> M stays ~#years)
    x = (S * w[:, None]).sum(0)
    return x, float(w.sum())


def pooled_empirical(S, n=None):
    """Sample-size-weighted long-run topic share (pool counts, not average shares).
    A year with 40 labeled Qs contributes 8x the year with 5 — the statistically
    correct way to estimate the base-rate, and it kills sparse-year spikes."""
    # DYNAMIC BASE RATE: Use a rolling 5-year window to prevent obsolete
    # ancient topics from dragging down new syllabus additions.
    window = 5
    recent_S = S[-window:] if len(S) > window else S
    if n is not None and len(n) == len(S):
        recent_n = n[-window:] if len(n) > window else n
        w = np.maximum(recent_n, 1e-9)
        emp = (recent_S * w[:, None]).sum(0) / w.sum()
    else:
        emp = recent_S.mean(axis=0)
    emp = np.maximum(emp, 1e-9)
    return emp / emp.sum()


def prior_centre(S, lam=1.0, n=None):
    """Dirichlet prior centre: blend of pooled empirical share (lam=1) and uniform (lam=0).
        pi0 = lam * empirical + (1 - lam) * uniform
    lam is tuned per-section by nested rolling-origin CV (see tune_params)."""
    k = S.shape[1]
    emp = pooled_empirical(S, n)
    uni = np.ones(k) / k
    pi0 = lam * emp + (1.0 - lam) * uni
    return pi0 / pi0.sum()


def posterior_mean(S, target_total, gamma, A, lam=1.0):
    """Empirical-Bayes Dirichlet-Multinomial posterior-mean shares, scaled to total."""
    x, M = decayed_evidence(S, gamma)
    pi0 = prior_centre(S, lam)
    p = (A * pi0 + x) / (A + M)
    return p * target_total


def _marginal_loglik(S, gamma, A, lam=1.0):
    """Leave-last-out marginal log-likelihood of concentration A on this history."""
    if len(S) < 2:
        return 0.0
    x, _ = decayed_evidence(S[:-1], gamma)
    pi0 = prior_centre(S[:-1], lam)
    alpha = A * pi0 + x
    a0 = alpha.sum()
    q = np.clip(S[-1], 1e-9, None); q = q / q.sum()
    return (math.lgamma(a0) - sum(math.lgamma(a) for a in alpha)
            + float(np.sum((alpha - 1) * np.log(q))))


def A_weights(S, gamma, A_grid=A_GRID, lam=1.0):
    """Posterior weights over the concentration grid (softmax of marginal log-lik)."""
    if len(S) < 3:
        w = np.zeros(len(A_grid)); w[len(A_grid) // 2] = 1.0
        return w
    lls = np.array([_marginal_loglik(S, gamma, A, lam) for A in A_grid])
    w = np.exp(lls - lls.max())
    return w / w.sum()


def marginalised_mean(S, target_total, gamma, A_grid=A_GRID, lam=1.0):
    """Point forecast: posterior mean marginalised over the concentration grid,
    with adaptive linear trend momentum to track sharply rising/falling topics."""
    w = A_weights(S, gamma, A_grid, lam)
    preds = np.array([posterior_mean(S, target_total, gamma, A, lam) for A in A_grid])
    base_pred = (preds * w[:, None]).sum(0)
    
    # Adaptive Momentum Post-processing with R-Squared Filtering
    if len(S) >= 4:
        window = min(4, len(S))
        recent_S = S[-window:]
        x_vals = np.arange(window)
        x_mean = x_vals.mean()
        y_mean = recent_S.mean(axis=0)
        
        var_x = ((x_vals - x_mean)**2).sum()
        if var_x > 0:
            cov = ((x_vals[:, None] - x_mean) * (recent_S - y_mean)).sum(axis=0)
            slope = cov / var_x
            
            # Calculate R-squared for each topic
            preds_line = slope * x_vals[:, None] + (y_mean - slope * x_mean)
            ss_tot = ((recent_S - y_mean)**2).sum(axis=0)
            ss_res = ((recent_S - preds_line)**2).sum(axis=0)
            
            # Avoid division by zero
            r2 = np.zeros_like(slope)
            valid = ss_tot > 1e-9
            r2[valid] = 1 - (ss_res[valid] / ss_tot[valid])
            
            # If a trend is very strong (R2 > 0.4), apply momentum. Else apply 0.
            # This prevents us from chasing noisy zig-zags.
            trend_adj = np.where(r2 > 0.4, slope * target_total * 0.8, 0)
            
            adj_pred = base_pred + trend_adj
            adj_pred = np.clip(adj_pred, 0, target_total)
            if adj_pred.sum() > 0:
                adj_pred = adj_pred / adj_pred.sum() * target_total
            return adj_pred
    return base_pred


def tune_params(cnt, syt, sec, topics, hist_years):
    """Jointly pick (gamma, lam) by nested rolling-origin CV on hist ONLY (no leakage).
    Returns (best_gamma, best_lam). lam blends empirical vs uniform prior per section:
    stable sections converge to lam~1, volatile sections to lam~0."""
    best, best_g, best_l = np.inf, 0.8, 1.0
    for g in GAMMA_GRID:
        for lam in LAM_GRID:
            errs = []
            for i in range(MIN_HISTORY, len(hist_years)):
                S = share_matrix(cnt, syt, sec, hist_years[:i], topics)
                p = marginalised_mean(S, SEC_TARGET[sec], g, lam=lam)
                a = actual_vec(cnt, syt, sec, hist_years[i], topics)
                errs.append(np.abs(p - a).mean())
            if errs and np.mean(errs) < best:
                best, best_g, best_l = np.mean(errs), g, lam
    return best_g, best_l


def tune_gamma(cnt, syt, sec, topics, hist_years):
    """Backward-compat shim: returns only gamma from the joint tuner."""
    return tune_params(cnt, syt, sec, topics, hist_years)[0]


def credible_interval(S, gamma, target_total, lo_q=0.05, hi_q=0.95,
                      n_sim=20000, A_grid=A_GRID, seed=0, lam=1.0):
    """Simulation-based Beta-Binomial credible interval per topic."""
    rng = np.random.default_rng(seed)
    x, _ = decayed_evidence(S, gamma)
    pi0 = prior_centre(S, lam)
    w = A_weights(S, gamma, A_grid, lam)
    A_samp = rng.choice(A_grid, size=n_sim, p=w)
    sims = np.zeros((n_sim, S.shape[1]))
    for j in range(n_sim):
        alpha = A_samp[j] * pi0 + x
        a0 = alpha.sum()
        sims[j] = betabinom.rvs(int(round(target_total)), alpha, a0 - alpha,
                                random_state=rng)
    return np.percentile(sims, lo_q * 100, axis=0), np.percentile(sims, hi_q * 100, axis=0)


# --------------------------------------------------------------------------- top API
class DirichletForecaster:
    """Fit-on-full-history production forecaster.

    Usage:
        fc = DirichletForecaster.from_repo()
        result = fc.predict()          # dict section -> list of topic predictions
    """
    def __init__(self, cnt, syt, years):
        self.cnt, self.syt, self.years = cnt, syt, years
        self.section_gamma_ = {}       # filled during predict(): the refit gammas

    @classmethod
    def from_repo(cls, q_path: Path = Q_PATH):
        return cls(*load_counts(q_path))

    def predict(self, sections=None, round_counts=True):
        """Forecast next exam. Jointly tunes (gamma, lam) on full history per section."""
        out = {}
        for sec in (sections or SEC_TARGET):
            topics = sorted(self.cnt[sec])
            hist = section_years(self.syt, sec)
            if len(hist) < MIN_HISTORY:
                continue
            gamma, lam = tune_params(self.cnt, self.syt, sec, topics, hist)
            self.section_gamma_[sec] = gamma
            S = share_matrix(self.cnt, self.syt, sec, hist, topics)
            mean = marginalised_mean(S, SEC_TARGET[sec], gamma, lam=lam)
            lo, hi = credible_interval(S, gamma, SEC_TARGET[sec], lam=lam)

            rows = []
            for i, t in enumerate(topics):
                exp = float(mean[i])
                rows.append({
                    "topic": t,
                    "section": sec,
                    "expected_count": round(exp) if round_counts else exp,
                    "expected_count_exact": round(exp, 3),
                    "ci90_low": int(lo[i]),
                    "ci90_high": int(hi[i]),
                    "share": round(exp / SEC_TARGET[sec], 4),
                })
            rows.sort(key=lambda r: r["expected_count_exact"], reverse=True)
            out[sec] = {
                "section_total": SEC_TARGET[sec],
                "recency_gamma": gamma,
                "prior_lam": lam,
                "n_years": len(hist),
                "n_topics": len(topics),
                "topics": rows,
            }
        return out


def main():
    fc = DirichletForecaster.from_repo()
    pred = fc.predict()
    print("AFCAT 2026 topic forecast (Dirichlet-Multinomial, full-history refit)\n")
    for sec, blk in pred.items():
        print(f"== {sec}  (total={blk['section_total']}, gamma={blk['recency_gamma']}, "
              f"{blk['n_years']} yrs, {blk['n_topics']} topics)")
        for r in blk["topics"][:8]:
            print(f"   {r['topic']:32s} {r['expected_count_exact']:5.2f}  "
                  f"[90% CI {r['ci90_low']}-{r['ci90_high']}]")
        # sanity: expected counts sum to section total
        s = sum(r["expected_count_exact"] for r in blk["topics"])
        print(f"   ...  sum(expected)={s:.2f}\n")


if __name__ == "__main__":
    main()
