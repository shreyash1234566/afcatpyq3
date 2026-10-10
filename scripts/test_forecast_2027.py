import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dirichlet_forecaster import DirichletForecaster

def allocate_exact(topics_list, section_target):
    allocs = {}
    remainders = []
    for row in topics_list:
        t = row["topic"]
        e = row["expected_count_exact"]
        allocs[t] = int(math.floor(e))
        remainders.append((e - math.floor(e), t))
    shortfall = section_target - sum(allocs.values())
    remainders.sort(reverse=True, key=lambda x: x[0])
    for i in range(shortfall):
        if i < len(remainders):
            allocs[remainders[i][1]] += 1
    return allocs

forecaster = DirichletForecaster.from_repo()
plan_2027 = forecaster.predict()

total_alloc = 0
for sec, blk in plan_2027.items():
    allocs = allocate_exact(blk["topics"], blk["section_total"])
    s_sum = sum(allocs.values())
    total_alloc += s_sum
    print(f"\n=== {sec} (Allocated: {s_sum}/{blk['section_total']}) ===")
    for t, n in sorted(allocs.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"  {t:<25}: {n} Qs")

print(f"\nTotal Paper Questions: {total_alloc} / 100")
