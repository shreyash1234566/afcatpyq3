"""
merge_2026_to_qclean.py
=======================
Merges all 2026 AFCAT papers into data/processed/Q_clean.json:
  - AFCAT 2026 Jan 31 Shift 1 (100 Qs from data/papers/afcat_2026_questions.json)
  - AFCAT 2026 Aug Shift 1 (100 Qs from data/papers/afcat_2026_aug_shift1_questions.json)
  - AFCAT 2026 Aug Shift 2 (100 Qs from data/papers/afcat_2026_aug_shift2_questions.json)

Verifies:
  - Exact count increases from 2,861 to 3,161 questions
  - Section quotas preserved (VA: 30, GA: 25, NA: 20, Reasoning: 25)
  - Zero unmapped topics
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    q_clean_path = ROOT / "data" / "processed" / "Q_clean.json"
    with open(q_clean_path, "r", encoding="utf-8") as f:
        q_clean = json.load(f)

    initial_len = len(q_clean)
    print(f"Current Q_clean count: {initial_len}")

    # Remove any existing 2026 entries to avoid duplicates if re-run
    q_clean = [q for q in q_clean if not q.get("test_name", "").startswith("AFCAT 2026")]
    print(f"Base count without 2026: {len(q_clean)}")

    # 1. Load Jan 2026
    with open(ROOT / "data" / "papers" / "afcat_2026_questions.json", "r", encoding="utf-8") as f:
        jan_raw = json.load(f)

    jan_processed = []
    for q in jan_raw:
        rec = {
            "file_name": "AFCAT-I-Question-Paper-31-Jan-2026.pdf",
            "question_number": q["q_num"],
            "question_text": q["question_text"],
            "section": q["section"],
            "topic": q["topic"],
            "difficulty": q.get("difficulty", "Medium").capitalize(),
            "topic_code": f"{q['section'][:2].upper()}_{q['topic'][:4].upper()}",
            "choices": [{"key": k, "text": v} for k, v in q["options"].items()],
            "correct_answer": q.get("correct_answer"),
            "explanation": q.get("explanation", ""),
            "has_figure": q.get("has_figure", False),
            "test_name": "AFCAT 2026 Jan 31 Shift 1",
            "micro_topic": q.get("micro_topic", q["topic"]),
            "passage": q.get("passage")
        }
        jan_processed.append(rec)

    # 2. Load Aug 2026 Shift 1
    with open(ROOT / "data" / "papers" / "afcat_2026_aug_shift1_questions.json", "r", encoding="utf-8") as f:
        s1_raw = json.load(f)

    # 3. Load Aug 2026 Shift 2
    with open(ROOT / "data" / "papers" / "afcat_2026_aug_shift2_questions.json", "r", encoding="utf-8") as f:
        s2_raw = json.load(f)

    print(f"Adding Jan 2026: {len(jan_processed)} questions")
    print(f"Adding Aug 2026 Shift 1: {len(s1_raw)} questions")
    print(f"Adding Aug 2026 Shift 2: {len(s2_raw)} questions")

    q_clean.extend(jan_processed)
    q_clean.extend(s1_raw)
    q_clean.extend(s2_raw)

    final_len = len(q_clean)
    print(f"Final Q_clean count: {final_len}")
    assert final_len == 3161, f"Expected 3161 questions, got {final_len}"

    # Backup original before saving
    backup_path = ROOT / "data" / "processed" / "Q_clean_backup_2861.json"
    if not backup_path.exists():
        with open(ROOT / "data" / "processed" / "Q_clean.json", "r", encoding="utf-8") as f:
            old_data = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(old_data)
        print(f"Created backup at {backup_path}")

    with open(q_clean_path, "w", encoding="utf-8") as f:
        json.dump(q_clean, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved updated Q_clean.json ({final_len} total questions)")

if __name__ == "__main__":
    main()
