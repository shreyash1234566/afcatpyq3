"""
inspect_2026_aug.py
Examines questions in Shift 1 and Shift 2 to verify section ranges, question types, and structure.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for shift_name, filename in [("Shift 1", "afcat_2026_aug_shift1_raw.json"), ("Shift 2", "afcat_2026_aug_shift2_raw.json")]:
    p = ROOT / "data" / "papers" / filename
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n=================== {shift_name} ({len(data)} Qs) ===================")
    for q in data:
        qn = q["question_number"]
        sec = q["section"]
        txt = q["question_text"]
        has_fig = q.get("has_figure", False)
        # print boundary questions and samples
        if qn in [1, 2, 5, 10, 15, 20, 25, 30, 31, 35, 40, 45, 50, 55, 56, 60, 65, 70, 75, 76, 80, 85, 90, 95, 100]:
            print(f"Q.{qn:2d} [{sec:17s}] fig={has_fig} | {txt[:80]}...")
