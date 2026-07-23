import json
from pathlib import Path

BASE = Path(r"e:\afcatpyq3")
Q_JSON_PATH = BASE / "data/processed/Q.json"
data = json.load(open(Q_JSON_PATH, encoding="utf-8"))

FIGURE_INDICATORS = [
    "diagram (a)", "diagram (b)", "figure (a)", "figure (b)",
    "following figure", "figure below", "given figure", "answer figure",
    "figure shown", "figure given", "question figure",
]

NV_TOPICS = {
    "Venn Diagrams", "Non-Verbal Series", "Non-Verbal Analogy",
    "Non-Verbal Classification", "Non-Verbal Pattern", "Non-Verbal Puzzle",
    "Non-Verbal Orientation", "Non-Verbal Spatial", "Pattern Completion",
}

# Reset all has_figure flags
for q in data:
    q["has_figure"] = False
    
# Re-apply strict logic
for q in data:
    q_text_lower = q.get("question_text", "").lower()
    choices = q.get("choices", [])
    choices_text = " ".join(
        (c.get("text","") if isinstance(c,dict) else str(c)).lower()
        for c in choices
    )
    combined = q_text_lower + " " + choices_text
    topic = q.get("topic", "")
    
    if any(ind in combined for ind in FIGURE_INDICATORS) or topic in NV_TOPICS:
        q["has_figure"] = True

fig_count = sum(1 for q in data if q.get("has_figure"))
print(f"Corrected has_figure count: {fig_count}")

# Save
json.dump(data, open(Q_JSON_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
