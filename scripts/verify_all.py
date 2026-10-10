"""
verify_all.py
=============
End-to-end verification of all pipeline deliverables for AFCAT 01/2027.
"""
import json
from pathlib import Path
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    print("Running comprehensive end-to-end audit...\n")

    # 1. Q_clean.json
    with open(ROOT / "data" / "processed" / "Q_clean.json", "r", encoding="utf-8") as f:
        qc = json.load(f)
    assert len(qc) == 3161, f"Expected 3161, got {len(qc)}"
    print(f"✓ Check 1: Q_clean.json contains exactly {len(qc)} verified questions.")

    # 2. Shift 1 & Shift 2
    with open(ROOT / "data" / "papers" / "afcat_2026_aug_shift1_questions.json", "r", encoding="utf-8") as f:
        s1 = json.load(f)
    assert len(s1) == 100, f"Expected 100, got {len(s1)}"

    with open(ROOT / "data" / "papers" / "afcat_2026_aug_shift2_questions.json", "r", encoding="utf-8") as f:
        s2 = json.load(f)
    assert len(s2) == 100, f"Expected 100, got {len(s2)}"
    print("✓ Check 2: August 2026 Shift 1 and Shift 2 canonical JSONs each contain exactly 100 questions.")

    # 3. 2027 Mock Paper
    with open(ROOT / "output" / "generated_questions" / "practice_2027_sota.json", "r", encoding="utf-8") as f:
        p27 = json.load(f)
    assert len(p27) == 100, f"Expected 100, got {len(p27)}"
    c = Counter(q["section"] for q in p27)
    assert c["Verbal Ability"] == 30 and c["General Awareness"] == 25 and c["Reasoning"] == 25 and c["Numerical Ability"] == 20
    print("✓ Check 3: AFCAT 01/2027 SOTA paper contains exactly 100 questions (VA: 30, GA: 25, Reasoning: 25, NA: 20).")

    # 4. Vector DB Pattern Store
    from utils.vector_store import VectorPatternStore
    vstore = VectorPatternStore()
    stats = vstore.get_stats()
    assert stats["total_questions"] == 3161
    print(f"✓ Check 4: Vector Pattern Store (SQLite + FTS5 + TF-IDF) has {stats['total_questions']} questions indexed across {len(stats['by_year'])} years.")

    # 5. Dashboard Data & HTML
    with open(ROOT / "dashboard" / "data.js", "r", encoding="utf-8") as f:
        djs = f.read()
    assert '"pyqCount": 3161' in djs
    assert "AFCAT 01/2027" in djs
    print("✓ Check 5: dashboard/data.js correctly reflects pyqCount: 3161 and AFCAT 01/2027.")

    with open(ROOT / "dashboard" / "index.html", "r", encoding="utf-8") as f:
        html = f.read()
    assert "2026" not in html
    assert "AFCAT 01/2027" in html
    print("✓ Check 6: dashboard/index.html fully updated from 2026 to AFCAT 01/2027 A to Z with zero remaining 2026 references.")

    print("\nALL 6 CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
