import sys, collections, numpy as np, math
from pathlib import Path
import json, re

Q_PATH = Path("e:/afcatpyq3/data/processed/Q.json")

def _year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

data = json.load(open(Q_PATH, encoding="utf-8"))
cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
syt = collections.defaultdict(collections.Counter)
SEC_TARGET = {"Verbal Ability": 30, "General Awareness": 25, "Reasoning": 25, "Numerical Ability": 20}

# Load the current normalizer to keep QA and GA the same, but bypass VA and Reasoning
try:
    sys.path.insert(0, "e:/afcatpyq3/scripts")
    from topic_normalization_map import TOPIC_NORMALIZATION
except ImportError:
    TOPIC_NORMALIZATION = {}

for q in data:
    y, s, t = _year(q.get("file_name")), q.get("section"), q.get("topic")
    if y and s in SEC_TARGET and t:
        if s in ["Verbal Ability", "Reasoning"]:
            pass # Keep raw granular topic!
        else:
            t = TOPIC_NORMALIZATION.get(t, t)
        cnt[s][t][y] += 1
        syt[s][y] += 1

years = sorted({y for s in syt for y in syt[s]})
MIN_SEC_QS = 5
MIN_HISTORY = 3

sys.path.insert(0, "e:/afcatpyq3/models")
from dirichlet_forecaster import tune_params, share_matrix, marginalised_mean, actual_vec, section_years

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
        print(f"{sec:20} Granular Topics={len(cnt[sec])} | MAE={np.mean(m):.3f} (naive={np.mean(n):.3f})")
