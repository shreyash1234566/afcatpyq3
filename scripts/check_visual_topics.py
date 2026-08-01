"""
Check Non-Verbal Pattern and Venn Diagrams questions from past papers
to understand what dummy/placeholder data we can use.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "data" / "processed" / "Q_clean.json").read_text(encoding="utf-8"))

def get_year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

VISUAL_TOPICS = ["Non-Verbal Pattern", "Venn Diagrams", "Non-Verbal Series",
                 "Spatial Ability", "Non-Verbal Classification", "Non-Verbal Analogy",
                 "Dot Situation"]

print("=== Visual/Figure Topic Questions (last 5 years) ===\n")
for topic in VISUAL_TOPICS:
    qs = [q for q in data if q.get("topic") == topic and (get_year(q.get("file_name","")) or 0) >= 2020]
    print(f"\n[{topic}] - {len(qs)} questions from 2020+")
    for q in qs[:3]:
        txt = q.get("question_text","")
        has_fig = q.get("has_figure", False)
        yr = get_year(q.get("file_name",""))
        print(f"  {yr} | has_figure={has_fig} | {txt[:120]}")
