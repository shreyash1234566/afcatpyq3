import sys, collections, numpy as np, math
from pathlib import Path

Q_PATH = Path("e:/afcatpyq3/data/processed/Q.json")
try:
    sys.path.insert(0, "e:/afcatpyq3/scripts")
    from topic_normalization_map import TOPIC_NORMALIZATION
except ImportError:
    TOPIC_NORMALIZATION = {}

import json, re
def _year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

def load_counts():
    data = json.load(open(Q_PATH, encoding="utf-8"))
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    syt = collections.defaultdict(collections.Counter)
    SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25, "Reasoning": 25, "Numerical Ability": 20}
    for q in data:
        y, s, t = _year(q.get("file_name")), q.get("section"), q.get("topic")
        if y and s in SEC_TARGET and t:
            t = TOPIC_NORMALIZATION.get(t, t)
            cnt[s][t][y] += 1
            syt[s][y] += 1
    years = sorted({y for s in syt for y in syt[s]})
    return cnt, syt, years

cnt, syt, years = load_counts()
SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25, "Reasoning": 25, "Numerical Ability": 20}
MIN_SEC_QS = 5
MIN_HISTORY = 3

GAMMA_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
A_GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12], float)

def section_years(syt, sec):
    return [y for y in sorted(syt[sec]) if syt[sec][y] >= MIN_SEC_QS]

def actual_vec(cnt, syt, sec, y, topics):
    tot = syt[sec][y]
    return np.array([cnt[sec][t][y] for t in topics], float) / tot * SEC_TARGET[sec]

def share_matrix(cnt, syt, sec, hist_years, topics):
    M = np.zeros((len(hist_years), len(topics)))
    for i, y in enumerate(hist_years):
        tot = syt[sec][y]
        if tot: M[i] = np.array([cnt[sec][t][y] for t in topics], float) / tot
    return M

def decayed_evidence(S, gamma, n=None):
    L = len(S) - 1
    w = np.array([gamma ** (L - k) for k in range(len(S))])
    if n is not None and len(n) == len(S):
        nn = np.maximum(n, 1e-9)
        w = w * (nn / nn.mean())
    x = (S * w[:, None]).sum(0)
    return x, float(w.sum())

def pooled_empirical(S, n=None):
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
    k = S.shape[1]
    emp = pooled_empirical(S, n)
    uni = np.ones(k) / k
    pi0 = lam * emp + (1.0 - lam) * uni
    return pi0 / pi0.sum()

def posterior_mean(S, target_total, gamma, A, lam=1.0):
    x, M = decayed_evidence(S, gamma)
    pi0 = prior_centre(S, lam)
    p = (A * pi0 + x) / (A + M)
    return p * target_total

def _marginal_loglik(S, gamma, A, lam=1.0):
    if len(S) < 2: return 0.0
    x, _ = decayed_evidence(S[:-1], gamma)
    pi0 = prior_centre(S[:-1], lam)
    alpha = A * pi0 + x
    a0 = alpha.sum()
    q = np.clip(S[-1], 1e-9, None); q = q / q.sum()
    return (math.lgamma(a0) - sum(math.lgamma(a) for a in alpha)
            + float(np.sum((alpha - 1) * np.log(q))))

def A_weights(S, gamma, A_grid=A_GRID, lam=1.0):
    if len(S) < 3:
        w = np.zeros(len(A_grid)); w[len(A_grid) // 2] = 1.0
        return w
    lls = np.array([_marginal_loglik(S, gamma, A, lam) for A in A_grid])
    w = np.exp(lls - lls.max())
    return w / w.sum()

def marginalised_mean(S, target_total, gamma, A_grid=A_GRID, lam=1.0):
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
            
            # If a trend is very strong (R2 > 0.5), apply momentum. Else apply 0.
            # This prevents us from chasing noisy zig-zags.
            trend_adj = np.where(r2 > 0.4, slope * target_total * 0.8, 0)
            
            adj_pred = base_pred + trend_adj
            adj_pred = np.clip(adj_pred, 0, target_total)
            if adj_pred.sum() > 0:
                adj_pred = adj_pred / adj_pred.sum() * target_total
            return adj_pred
            
    return base_pred

def tune_params(cnt, syt, sec, topics, hist_years):
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

# --- EVALUATION ---

rows = []
for sec in ["Reasoning", "Verbal Ability"]:
    topics = sorted(cnt[sec])
    yrs = section_years(syt, sec)
    for i in range(MIN_HISTORY, len(yrs)):
        hist, ty = yrs[:i], yrs[i]
        g, lam = tune_params(cnt, syt, sec, topics, hist)
        S = share_matrix(cnt, syt, sec, hist, topics)
        pred = marginalised_mean(S, SEC_TARGET[sec], g, lam=lam)
        act = actual_vec(cnt, syt, sec, ty, topics)
        naive = actual_vec(cnt, syt, sec, hist[-1], topics)
        mae = float(np.abs(pred - act).mean())
        nmae = float(np.abs(naive - act).mean())
        rows.append((ty, sec, mae, nmae))

for sec in ["Reasoning", "Verbal Ability"]:
    sr = [(r[2], r[3]) for r in rows if r[1] == sec]
    if sr:
        m, n = zip(*sr)
        print(f"{sec:20} MAE={np.mean(m):.3f}  naiveMAE={np.mean(n):.3f} MASE={np.mean(m)/np.mean(n):.3f}")
