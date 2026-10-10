"""
check_images_and_taxonomy.py
Comprehensive audit of:
1. Physical images for all visual questions in S1 and S2
2. Canonical taxonomy validation against data/topic_map.json and existing Q_clean.json
3. How images are loaded in dashboard/index.html (URL resolution)
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 80)
    print("AUDIT PART 1: PHYSICAL FIGURE IMAGES & DASHBOARD COMPATIBILITY")
    print("=" * 80)

    with open(ROOT / "data" / "papers" / "afcat_2026_aug_shift1_questions.json", "r", encoding="utf-8") as f:
        s1 = json.load(f)
    with open(ROOT / "data" / "papers" / "afcat_2026_aug_shift2_questions.json", "r", encoding="utf-8") as f:
        s2 = json.load(f)

    missing = []
    present = 0

    for name, qs in [("August 2026 Shift 1", s1), ("August 2026 Shift 2", s2)]:
        for q in qs:
            if q.get("has_figure"):
                ip = q.get("image_path")
                paths = [ip] if isinstance(ip, str) else (ip if isinstance(ip, list) else [])
                for p in paths:
                    local_p = ROOT / p
                    if local_p.exists() and local_p.stat().st_size > 0:
                        present += 1
                        print(f"  [FOUND] {name} Q.{q['question_number']}: {p} ({local_p.stat().st_size:,} bytes)")
                    else:
                        missing.append((name, q["question_number"], p))
                        print(f"  [MISSING] {name} Q.{q['question_number']}: {p}")

    print(f"\nImage Summary: {present} files present, {len(missing)} missing.")

    print("\n" + "=" * 80)
    print("AUDIT PART 2: TAXONOMY CONSISTENCY WITH data/topic_map.json")
    print("=" * 80)

    with open(ROOT / "data" / "topic_map.json", "r", encoding="utf-8") as f:
        topic_map = json.load(f)

    # Flatten topic_map to valid sections and topics
    valid_taxonomy = {}
    for sec, chapters in topic_map.items():
        valid_taxonomy[sec] = set(chapters.keys())

    all_valid_topics = set()
    for sec, tops in valid_taxonomy.items():
        all_valid_topics.update(tops)

    print(f"Canonical taxonomy has {len(valid_taxonomy)} sections and {len(all_valid_topics)} topics.")

    # Check S1 and S2
    taxonomy_errors = []
    for name, qs in [("August 2026 Shift 1", s1), ("August 2026 Shift 2", s2)]:
        for q in qs:
            sec = q.get("section")
            top = q.get("topic")
            micro = q.get("micro_topic")
            qnum = q.get("question_number")

            if sec not in valid_taxonomy:
                taxonomy_errors.append((name, qnum, f"Invalid section: '{sec}'"))
            elif top not in valid_taxonomy[sec]:
                taxonomy_errors.append((name, qnum, f"Topic '{top}' not valid in section '{sec}'"))

            if not micro or not isinstance(micro, str) or len(micro.strip()) < 2:
                taxonomy_errors.append((name, qnum, f"Empty or invalid micro_topic: '{micro}'"))

    if taxonomy_errors:
        print(f"FAILED: Found {len(taxonomy_errors)} taxonomy errors:")
        for err in taxonomy_errors:
            print(f"  - {err[0]} Q.{err[1]}: {err[2]}")
    else:
        print("PERFECT: 100% of the 200 questions match the canonical topic_map.json taxonomy!")

    print("\n" + "=" * 80)
    print("AUDIT PART 3: MICRO-TOPIC DISTRIBUTION & VOCABULARY AUDIT")
    print("=" * 80)

    # Load historic Q_clean questions to inspect micro-topics
    with open(ROOT / "data" / "processed" / "Q_clean.json", "r", encoding="utf-8") as f:
        q_clean = json.load(f)

    # Historical micro-topics per topic
    hist_micros = defaultdict(set)
    for q in q_clean[:2861]:  # historical before 2026 August
        t = q.get("topic")
        m = q.get("micro_topic")
        if t and m:
            hist_micros[t].add(m.lower().strip())

    s1_s2_micros = Counter()
    novel_micros = []
    matched_micros = 0

    for name, qs in [("Shift 1", s1), ("Shift 2", s2)]:
        for q in qs:
            t = q.get("topic")
            m = q.get("micro_topic", "").strip()
            s1_s2_micros[m] += 1
            if m.lower() in hist_micros[t]:
                matched_micros += 1
            else:
                novel_micros.append((name, q["question_number"], t, m, q.get("question_text", "")[:70]))

    print(f"Total micro-topic assignments: {len(s1) + len(s2)}")
    print(f"Direct match with historical Q_clean conventions: {matched_micros} / 200 ({matched_micros/2:.1f}%)")
    print(f"Novel granular sub-topics introduced: {len(novel_micros)} / 200")
    if novel_micros:
        print("\nReviewing novel granular sub-topics to verify they are accurate (not nonsense/hallucinations):")
        for nov in novel_micros[:15]:
            print(f"  [{nov[0]} Q.{nov[1]}] Topic: {nov[2]} | Micro: '{nov[3]}' | Text: {nov[4]}...")

    print("\n" + "=" * 80)
    print("AUDIT PART 4: SECTION BALANCE PER SHIFT")
    print("=" * 80)
    for name, qs in [("Shift 1", s1), ("Shift 2", s2)]:
        c = Counter(q["section"] for q in qs)
        print(f"  {name}: Verbal={c['Verbal Ability']} (target 30), GA={c['General Awareness']} (target 25), Reasoning={c['Reasoning']} (target 25), Numerical={c['Numerical Ability']} (target 20) | Total={sum(c.values())}")

    print("\n" + "=" * 80)
    print("AUDIT PART 5: DASHBOARD IMAGE RENDERING LOGIC")
    print("=" * 80)
    with open(ROOT / "dashboard" / "index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Look for image handling in index.html
    import re
    img_logic = re.search(r"finalImgSrc\s*=\s*(.+?);", html)
    if img_logic:
        print("  Found image src resolution in dashboard/index.html:")
        print(f"  {img_logic.group(0)}")
    else:
        print("  WARNING: Could not find finalImgSrc in index.html!")

if __name__ == "__main__":
    main()
