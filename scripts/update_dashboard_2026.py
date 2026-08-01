"""
Run only the dashboard update from the already-saved practice_2026_august.json
"""
import json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dirichlet_forecaster import DirichletForecaster

# Load already-generated questions (August Paper 2 Mock)
q_path = ROOT / "output" / "generated_questions" / "practice_2026_august.json"
if not q_path.exists():
    print(f"Error: {q_path} not found. Generate the paper first.")
    sys.exit(1)

questions = json.loads(q_path.read_text(encoding="utf-8"))
print(f"Loaded {len(questions)} questions from {q_path.name}")

# Re-run DM to get the base plan
model = DirichletForecaster.from_repo()
plan_2026 = model.predict()

# Apply Hybrid P1-Weighted Plan (70% P1 / 30% DM)
print("Applying 70% Paper 1 / 30% DM hybrid weighting...")
p1_path = ROOT / "data" / "papers" / "afcat_2026_questions.json"
p1_data = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else []

p1_counts = defaultdict(int)
for q in p1_data:
    p1_counts[(q.get("section"), q.get("topic"))] += 1

for sec, blk in plan_2026.items():
    for t_row in blk["topics"]:
        topic = t_row["topic"]
        dm_exp = t_row["expected_count_exact"]
        p1_exp = p1_counts.get((sec, topic), 0)
        
        hybrid_exp = 0.70 * p1_exp + 0.30 * dm_exp
        
        t_row["expected_count_exact"] = hybrid_exp
        t_row["dm_expected"] = dm_exp
        t_row["p1_actual"] = p1_exp

    blk["topics"].sort(key=lambda x: -x["expected_count_exact"])

section_stats = {sec: {"total": blk["section_total"], "topics": blk["topics"]}
                 for sec, blk in plan_2026.items()}

# Import and run dashboard update
from scripts.generate_2026_sota import _update_dashboard
_update_dashboard(questions, plan_2026, section_stats)

print("[DONE] dashboard/data.js updated successfully!")
print(f"  Questions: {len(questions)}")

# Quick summary
from collections import Counter
by_sec = Counter(q["section"] for q in questions)
for sec, cnt in by_sec.items():
    print(f"  {sec}: {cnt} questions")
