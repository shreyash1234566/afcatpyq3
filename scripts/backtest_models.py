"""
backtest_models.py — Honest, reproducible model comparison for AFCAT topic forecasting.

This is the *proof* file. It answers one question with no leakage:
    "Which forecaster best predicts next year's per-topic question distribution,
     given only the years strictly before it?"

Protocol:
  - Rolling-origin evaluation (expanding window): predict year t using years < t.
  - NESTED hyperparameter tuning: for every test year t, the model's hyperparameters
    are chosen using ONLY data < t (an inner rolling-origin loop). No test fold ever
    informs the hyperparameters that score it. This gives an honest generalization
    number, not an optimistic grid-search-against-the-test-set number.
  - Metric: MAE in "questions per topic" on the section-scaled distribution, plus
    MASE (vs naive last-year) and cosine similarity of the predicted vs actual
    topic distribution.

Models compared:
  naive_last     : predict last observed year's distribution (persistence baseline)
  mean_all       : historical mean distribution
  ewma           : recency-weighted mean (fixed alpha)
  dirichlet      : symmetric Dirichlet-Multinomial, recency-decayed evidence,
                   concentration marginalised over a grid weighted by marginal
                   likelihood (Bayesian "infer the concentration" — closed form)
  xgboost/rf     : the *current* project approach (global feature-based regressors)

Run:  python scripts/backtest_models.py
"""
import json, re, collections, warnings, math, sys
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
Q_PATH = ROOT / "data" / "processed" / "Q.json"
SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25,
              "Reasoning": 25, "Numerical Ability": 20}
MIN_SEC_QS = 5          # a year counts for a section only if it has >= this many Qs
MIN_HISTORY = 3         # need at least this many prior years to make a forecast


# ----------------------------------------------------------------------------- data
def load_counts():
    data = json.load(open(Q_PATH, encoding="utf-8"))
    yr = lambda fn: (int(m.group(1)) if (m := re.search(r"(20\d\d)", fn or "")) else None)
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    syt = collections.defaultdict(collections.Counter)
    for q in data:
        y, s, t = yr(q.get("file_name")), q.get("section"), q.get("topic")
        if y and s in SEC_TARGET and t:
            cnt[s][t][y] += 1
            syt[s][y] += 1
    years = sorted({y for s in syt for y in syt[s]})
    return cnt, syt, years


def section_years(syt, sec):
    return [y for y in sorted(syt[sec]) if syt[sec][y] >= MIN_SEC_QS]


def actual_vec(cnt, syt, sec, y, topics):
    """Actual topic distribution for one year, scaled to the section's fixed size."""
    tot = syt[sec][y]
    return np.array([cnt[sec][t][y] for t in topics], float) / tot * SEC_TARGET[sec]


def share_matrix(cnt, syt, sec, hist_years, topics):
    """Per-year within-section topic SHARES (each row sums to 1). Equal weight per year
    regardless of how many papers that year had."""
    M = np.zeros((len(hist_years), len(topics)))
    for i, y in enumerate(hist_years):
        tot = syt[sec][y]
        if tot:
            M[i] = np.array([cnt[sec][t][y] for t in topics], float) / tot
    return M


# ------------------------------------------------------------------------- forecasters
def f_naive_last(S, target_total):
    p = S[-1]
    return p / p.sum() * target_total


def f_mean_all(S, target_total):
    p = S.mean(0)
    return p / p.sum() * target_total


def f_ewma(S, target_total, alpha):
    w = np.array([(1 - alpha) ** (len(S) - 1 - k) for k in range(len(S))])
    w /= w.sum()
    p = np.average(S, axis=0, weights=w)
    return p / p.sum() * target_total


def decayed_evidence(S, gamma):
    """Recency-decayed sufficient statistic. Rows are yearly shares (sum 1 each).
    Returns x (per-topic decayed mass) with sum(x) = M = effective #years."""
    L = len(S) - 1
    w = np.array([gamma ** (L - k) for k in range(len(S))])
    x = (S * w[:, None]).sum(0)          # per-topic decayed share-mass
    return x, w.sum()                    # sum(x) == w.sum() == M


def dirichlet_posterior_mean(S, target_total, gamma, A):
    """Symmetric Dirichlet-Multinomial posterior mean shares -> scaled counts.
       p_i = (A/k + x_i) / (A + M).  w_uniform = A/(A+M)."""
    x, M = decayed_evidence(S, gamma)
    k = S.shape[1]
    p = (A / k + x) / (A + M)
    return p * target_total


def dirichlet_marginal_loglik(S, gamma, A):
    """Leave-last-out marginal likelihood of the concentration A on this history:
    how well does the DM fit built from years[:-1] predict the last year's shares?
    Used to weight A values (Bayesian marginalisation, closed-form proxy)."""
    if len(S) < 2:
        return 0.0
    x, M = decayed_evidence(S[:-1], gamma)
    k = S.shape[1]
    alpha = A / k + x                      # posterior Dirichlet params from history
    a0 = alpha.sum()
    last = S[-1]
    # Dirichlet log-density of the observed last-year share vector (smoothed)
    q = np.clip(last, 1e-9, None); q /= q.sum()
    ll = (math.lgamma(a0) - np.sum([math.lgamma(a) for a in alpha])
          + np.sum((alpha - 1) * np.log(q)))
    return ll


def f_dirichlet_marginalised(S, target_total, gamma, A_grid):
    """Marginalise the concentration over a grid, weighting each A by its marginal
    likelihood on the history (softmax of log-lik). No single plugged-in A."""
    if len(S) < 3:
        return dirichlet_posterior_mean(S, target_total, gamma, A_grid[len(A_grid)//2])
    lls = np.array([dirichlet_marginal_loglik(S, gamma, A) for A in A_grid])
    wts = np.exp(lls - lls.max()); wts /= wts.sum()
    preds = np.array([dirichlet_posterior_mean(S, target_total, gamma, A) for A in A_grid])
    return (preds * wts[:, None]).sum(0)


# --------------------------------------------------------------- nested tuning wrapper
DIR_GRID = [(g, A) for g in (0.6, 0.7, 0.8, 0.9, 1.0)
            for A in (0.25, 0.5, 1, 2, 3, 4, 6, 8, 12)]
EWMA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def inner_score(cnt, syt, sec, topics, train_years, make_pred):
    """Rolling-origin MAE of `make_pred` restricted to `train_years` only."""
    errs = []
    for i in range(MIN_HISTORY, len(train_years)):
        S = share_matrix(cnt, syt, sec, train_years[:i], topics)
        pred = make_pred(S, SEC_TARGET[sec])
        act = actual_vec(cnt, syt, sec, train_years[i], topics)
        errs.append(np.abs(pred - act).mean())
    return np.mean(errs) if errs else np.inf


def tune_dirichlet(cnt, syt, sec, topics, train_years):
    best, bestp = np.inf, DIR_GRID[0]
    for (g, A) in DIR_GRID:
        s = inner_score(cnt, syt, sec, topics, train_years,
                        lambda S, tt, g=g, A=A: dirichlet_posterior_mean(S, tt, g, A))
        if s < best:
            best, bestp = s, (g, A)
    return bestp


def tune_ewma(cnt, syt, sec, topics, train_years):
    best, besta = np.inf, EWMA_GRID[0]
    for a in EWMA_GRID:
        s = inner_score(cnt, syt, sec, topics, train_years,
                        lambda S, tt, a=a: f_ewma(S, tt, a))
        if s < best:
            best, besta = s, a
    return besta


# ---------------------------------------------------------------- optional current-approach
def global_ml_predict(cnt, syt, sec, topics, hist_years, kind):
    """The CURRENT project approach: featurise each topic's history and fit a global
    regressor across all topics, predict next-year count. Rescaled to section total."""
    try:
        if kind == "xgb":
            import xgboost as xgb
            Model = lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, verbosity=0)
        else:
            from sklearn.ensemble import RandomForestRegressor
            Model = lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    except Exception:
        return None

    def feats(series, upto):
        h = np.array(series[:upto], float)
        if len(h) == 0: h = np.array([0.0])
        last5 = h[-5:]
        slope = (np.polyfit(np.arange(len(h)), h, 1)[0] if len(h) >= 2 else 0.0)
        cons = 0
        for v in h[::-1]:
            if v > 0: cons += 1
            else: break
        nz = np.nonzero(h)[0]
        ysl = (len(h) - 1 - nz[-1]) if len(nz) else len(h)
        return [h.mean(), h[-2:].mean(), slope, ysl, cons, h.std(), h.max(), h.min(),
                *[(last5[i] if i < len(last5) else 0.0) for i in range(5)]]

    # scaled per-year counts per topic
    ser = {t: [actual_vec(cnt, syt, sec, y, topics)[k] for y in hist_years]
           for k, t in enumerate(topics)}
    X, Y = [], []
    for j in range(1, len(hist_years)):
        for t in topics:
            X.append(feats(ser[t], j)); Y.append(ser[t][j])
    if not X: return None
    m = Model(); m.fit(np.array(X), np.array(Y))
    Xt = np.array([feats(ser[t], len(hist_years)) for t in topics])
    p = np.clip(m.predict(Xt), 0, None)
    return p / p.sum() * SEC_TARGET[sec] if p.sum() > 0 else p


# ------------------------------------------------------------------------------- main
def main(run_ml=True):
    cnt, syt, years = load_counts()
    print("=" * 78)
    print("AFCAT topic-distribution forecasting — honest rolling-origin backtest")
    print("Nested hyperparameter tuning (no test fold informs its own hyperparameters)")
    print("=" * 78)

    agg = collections.defaultdict(list)          # model -> list of per-fold MAE
    cos_agg = collections.defaultdict(list)
    per_sec = collections.defaultdict(dict)

    for sec in SEC_TARGET:
        topics = sorted(cnt[sec])
        yrs = section_years(syt, sec)
        sec_err = collections.defaultdict(list)
        for i in range(MIN_HISTORY, len(yrs)):
            hist, ty = yrs[:i], yrs[i]
            S = share_matrix(cnt, syt, sec, hist, topics)
            act = actual_vec(cnt, syt, sec, ty, topics)

            preds = {}
            preds["naive_last"] = f_naive_last(S, SEC_TARGET[sec])
            preds["mean_all"] = f_mean_all(S, SEC_TARGET[sec])
            # nested-tuned models: choose hyperparams on hist ONLY
            a = tune_ewma(cnt, syt, sec, topics, hist)
            preds["ewma_nested"] = f_ewma(S, SEC_TARGET[sec], a)
            g, A = tune_dirichlet(cnt, syt, sec, topics, hist)
            preds["dirichlet_nested"] = dirichlet_posterior_mean(S, SEC_TARGET[sec], g, A)
            preds["dirichlet_marg"] = f_dirichlet_marginalised(
                S, SEC_TARGET[sec], gamma=0.8, A_grid=[0.25,0.5,1,2,3,4,6,8,12])
            if run_ml:
                for kind, name in (("xgb", "xgboost_current"), ("rf", "randomforest_current")):
                    p = global_ml_predict(cnt, syt, sec, topics, hist, kind)
                    if p is not None:
                        preds[name] = p

            for name, p in preds.items():
                mae = np.abs(p - act).mean()
                sec_err[name].append(mae)
                agg[name].append(mae)
                denom = np.linalg.norm(p) * np.linalg.norm(act)
                cos_agg[name].append(np.dot(p, act) / denom if denom else 0.0)

        for name, es in sec_err.items():
            per_sec[sec][name] = np.mean(es)

    # ---- report
    names = sorted(agg, key=lambda n: np.mean(agg[n]))
    naive = np.mean(agg["naive_last"])
    print(f"\n{'model':24s} {'MAE':>7s} {'MASE':>7s} {'cosine':>7s}   (lower MAE/MASE better)")
    print("-" * 60)
    for n in names:
        mae = np.mean(agg[n]); mase = mae / naive; cos = np.mean(cos_agg[n])
        star = "  <== BEST" if n == names[0] else ("  (current)" if "current" in n else "")
        print(f"{n:24s} {mae:7.3f} {mase:7.3f} {cos:7.3f}{star}")

    print("\nPer-section MAE:")
    hdr = ["naive_last", "ewma_nested", "dirichlet_nested", "dirichlet_marg"]
    hdr += [n for n in ("xgboost_current", "randomforest_current") if n in agg]
    print(f"{'section':20s} " + " ".join(f"{h[:14]:>15s}" for h in hdr))
    for sec in SEC_TARGET:
        row = " ".join(f"{per_sec[sec].get(h, float('nan')):15.3f}" for h in hdr)
        print(f"{sec:20s} {row}")

    best = names[0]
    print(f"\nBEST MODEL: {best}  (MAE={np.mean(agg[best]):.3f}, "
          f"{100*(1-np.mean(agg[best])/np.mean(agg['xgboost_current'])):.0f}% lower "
          f"error than current XGBoost)" if "xgboost_current" in agg else
          f"\nBEST MODEL: {best}  (MAE={np.mean(agg[best]):.3f})")
    return agg, per_sec


if __name__ == "__main__":
    main(run_ml="--no-ml" not in sys.argv)
