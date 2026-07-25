import sys, json, collections
from pathlib import Path

Q_PATH = Path("e:/afcatpyq3/data/processed/Q.json")
try:
    sys.path.insert(0, "e:/afcatpyq3/scripts")
    from topic_normalization_map import TOPIC_NORMALIZATION
except ImportError:
    TOPIC_NORMALIZATION = {}

data = json.load(open(Q_PATH, encoding="utf-8"))
import re

def _year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
syt = collections.defaultdict(collections.Counter)
for q in data:
    y = _year(q.get("file_name"))
    s = q.get("section")
    t = q.get("topic")
    if y and s and t:
        t = TOPIC_NORMALIZATION.get(t, t)
        cnt[s][t][y] += 1
        syt[s][y] += 1

years = sorted({y for s in syt for y in syt[s]})

for sec in ["Verbal Ability", "Reasoning"]:
    print(f"=== {sec} ===")
    topics = sorted(cnt[sec].keys(), key=lambda t: -sum(cnt[sec][t].values()))
    
    # Print header
    header = f"{'Topic':<25}" + "".join([f"{y:>5}" for y in years])
    print(header)
    print("-" * len(header))
    
    for t in topics:
        row = f"{t[:25]:<25}"
        for y in years:
            row += f"{cnt[sec][t][y]:>5}"
        print(row)
    print("\n")
