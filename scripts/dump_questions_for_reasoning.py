"""
dump_questions_for_reasoning.py
Dumps Shift 1 and Shift 2 questions by section so each can be inspected and reasoned through.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for shift, fname in [("s1", "afcat_2026_aug_shift1_raw.json"), ("s2", "afcat_2026_aug_shift2_raw.json")]:
    with open(ROOT / "data" / "papers" / fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    out_txt = ROOT / "data" / "papers" / f"{shift}_full_dump.txt"
    with open(out_txt, "w", encoding="utf-8") as out:
        for q in data:
            out.write(f"ID: Q.{q['question_number']} | Section: {q['section']} | Fig: {q.get('has_figure', False)} | Ans: {q['correct_answer']}\n")
            out.write(f"Stem: {q['question_text']}\n")
            for k in ["A", "B", "C", "D"]:
                out.write(f"  [{k}] {q['options'].get(k, '')}\n")
            if q.get("all_images"):
                out.write(f"  Images: {q['all_images']}\n")
            out.write("-" * 80 + "\n")
    print(f"Dumped {shift}: {out_txt}")
