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
import json, re, collections, math
from pathlib import Path
import numpy as np
from scipy.stats import betabinom, dirichlet as _dir  # dirichlet imported for completeness

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent.parent
Q_PATH = ROOT / "data" / "processed" / "Q.json"

# AFCAT 2026 fixed section sizes (known constants, NOT forecast).
SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25,
              "Reasoning": 25, "Numerical Ability": 20}
MIN_SEC_QS = 5          # a year is usable for a section only if it has >= this many Qs
MIN_HISTORY = 3         # minimum prior years before we trust a forecast
GAMMA_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
A_GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12], float)


# ----------------------------------------------------------------------------- data
def _year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None


def load_counts(q_path: Path = Q_PATH):
    """Return (cnt, syt, years): cnt[sec][topic][year]->int, syt[sec][year]->int."""
    data = json.load(open(q_path, encoding="utf-8"))
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    syt = collections.defaultdict(collections.Counter)
    for q in data:
        y, s, t = _year(q.get("file_name")), q.get("section"), q.get("topic")
        if y and s in SEC_TARGET and t:
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


# ------------------------------------------------------------------- core estimator
def decayed_evidence(S, gamma):
    """Recency-decayed sufficient statistic from a share-matrix S.
    Returns (x, M): x[i] = decayed share-mass for topic i, M = sum(weights) = eff #years."""
    L = len(S) - 1
    w = np.array([gamma ** (L - k) for k in range(len(S))])
    x = (S * w[:, None]).sum(0)
    return x, float(w.sum())


def posterior_mean(S, target_total, gamma, A):
    """Symmetric Dirichlet-Multinomial posterior-mean shares, scaled to section total."""
    x, M = decayed_evidence(S, gamma)
    k = S.shape[1]
    p = (A / k + x) / (A + M)
    return p * target_total


def _marginal_loglik(S, gamma, A):
    """Leave-last-out marginal log-likelihood of concentration A on this history."""
    if len(S) < 2:
        return 0.0
    x, _ = decayed_evidence(S[:-1], gamma)
    k = S.shape[1]
    alpha = A / k + x
    a0 = alpha.sum()
    q = np.clip(S[-1], 1e-9, None); q = q / q.sum()
    return (math.lgamma(a0) - sum(math.lgamma(a) for a in alpha)
            + float(np.sum((alpha - 1) * np.log(q))))


def A_weights(S, gamma, A_grid=A_GRID):
    """Posterior weights over the concentration grid (softmax of marginal log-lik)."""
    if len(S) < 3:
        w = np.zeros(len(A_grid)); w[len(A_grid) // 2] = 1.0
        return w
    lls = np.array([_marginal_loglik(S, gamma, A) for A in A_grid])
    w = np.exp(lls - lls.max())
    return w / w.sum()


def marginalised_mean(S, target_total, gamma, A_grid=A_GRID):
    """Point forecast: posterior mean marginalised over the concentration grid."""
    w = A_weights(S, gamma, A_grid)
    preds = np.array([posterior_mean(S, target_total, gamma, A) for A in A_grid])
    return (preds * w[:, None]).sum(0)


def tune_gamma(cnt, syt, sec, topics, hist_years):
    """Nested pick of recency gamma using an inner rolling-origin loop on hist ONLY."""
    best, best_g = np.inf, 0.8
    for g in GAMMA_GRID:
        errs = []
        for i in range(MIN_HISTORY, len(hist_years)):
            S = share_matrix(cnt, syt, sec, hist_years[:i], topics)
            p = marginalised_mean(S, SEC_TARGET[sec], g)
            a = actual_vec(cnt, syt, sec, hist_years[i], topics)
            errs.append(np.abs(p - a).mean())
        if errs and np.mean(errs) < best:
            best, best_g = np.mean(errs), g
    return best_g


def credible_interval(S, gamma, target_total, lo_q=0.05, hi_q=0.95,
                      n_sim=20000, A_grid=A_GRID, seed=0):
    """Simulation-based Beta-Binomial interval per topic, propagating A-uncertainty.
    Fixes the two undercoverage bugs: (1) marginalise A instead of plugging a fixed
    value; (2) use Beta-Binomial(N,...) — the correct count predictive — not a
    scaled Beta (which understates variance by factor 1+A/N)."""
    rng = np.random.default_rng(seed)
    x, _ = decayed_evidence(S, gamma)
    k = S.shape[1]
    w = A_weights(S, gamma, A_grid)
    A_samp = rng.choice(A_grid, size=n_sim, p=w)
    sims = np.zeros((n_sim, k))
    for j in range(n_sim):
        alpha = A_samp[j] / k + x
        a0 = alpha.sum()
        sims[j] = betabinom.rvs(int(round(target_total)), alpha, a0 - alpha,
                                random_state=rng)
    lo = np.percentile(sims, lo_q * 100, axis=0)
    hi = np.percentile(sims, hi_q * 100, axis=0)
    return lo, hi


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
        """Forecast next exam. Refits gamma + A-weights on the FULL history per section
        (nothing reused from any backtest fold)."""
        out = {}
        for sec in (sections or SEC_TARGET):
            topics = sorted(self.cnt[sec])
            hist = section_years(self.syt, sec)
            if len(hist) < MIN_HISTORY:
                continue
            # --- production refit on complete history ---
            gamma = tune_gamma(self.cnt, self.syt, sec, topics, hist)
            self.section_gamma_[sec] = gamma
            S = share_matrix(self.cnt, self.syt, sec, hist, topics)
            mean = marginalised_mean(S, SEC_TARGET[sec], gamma)
            lo, hi = credible_interval(S, gamma, SEC_TARGET[sec])

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
            # sort most-likely first; expected shares already sum to section total
            rows.sort(key=lambda r: r["expected_count_exact"], reverse=True)
            out[sec] = {
                "section_total": SEC_TARGET[sec],
                "recency_gamma": gamma,
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
