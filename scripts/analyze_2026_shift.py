"""
Analyze what changed in 2026 Paper 1 vs historical patterns.
This reveals the NEW exam pattern that Paper 2 will likely follow.
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
history = json.loads((ROOT / "data" / "processed" / "Q_clean.json").read_text(encoding="utf-8"))
paper_2026 = json.loads((ROOT / "data" / "papers" / "afcat_2026_questions.json").read_text(encoding="utf-8"))

def get_year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

# ── Historical average (2022-2025) per topic ──
recent_years = {2022, 2023, 2024, 2025}
recent_qs = [q for q in history if get_year(q.get("file_name", "")) in recent_years]

hist_topic_avg = defaultdict(float)
hist_section_avg = defaultdict(float)
for yr in recent_years:
    yr_qs = [q for q in recent_qs if get_year(q.get("file_name", "")) == yr]
    for q in yr_qs:
        hist_topic_avg[(q["section"], q["topic"])] += 1
    for q in yr_qs:
        hist_section_avg[q["section"]] += 1

# Average per paper (divide by number of papers in those years)
n_papers = len(recent_years)
for k in hist_topic_avg:
    hist_topic_avg[k] /= n_papers
for k in hist_section_avg:
    hist_section_avg[k] /= n_papers

# ── 2026 Paper 1 distribution ──
p1_topics = defaultdict(int)
p1_sections = defaultdict(int)
for q in paper_2026:
    p1_topics[(q["section"], q["topic"])] += 1
    p1_sections[q["section"]] += 1

# ── Compare ──
print("=" * 80)
print("PATTERN SHIFT ANALYSIS: 2026 Paper 1 vs Historical Average (2022-2025)")
print("=" * 80)

print(f"\n{'Section':<25} {'Hist Avg':>10} {'2026 P1':>10} {'Shift':>10}")
print("-" * 60)
for sec in ["Verbal Ability", "General Awareness", "Reasoning", "Numerical Ability"]:
    avg = hist_section_avg.get(sec, 0)
    actual = p1_sections.get(sec, 0)
    shift = actual - avg
    arrow = ">>>" if abs(shift) > 3 else (">" if shift > 0 else ("<" if shift < 0 else "="))
    print(f"  {sec:<23} {avg:>10.1f} {actual:>10} {shift:>+10.1f} {arrow}")

print(f"\n\n{'Topic':<35} {'Hist Avg':>10} {'2026 P1':>10} {'Shift':>10} {'Signal':<15}")
print("-" * 85)

all_keys = sorted(set(list(hist_topic_avg.keys()) + list(p1_topics.keys())))
big_shifts = []
for key in all_keys:
    avg = hist_topic_avg.get(key, 0)
    actual = p1_topics.get(key, 0)
    shift = actual - avg
    
    if abs(shift) >= 2:
        signal = "*** BIG SHIFT ***"
        big_shifts.append((key, avg, actual, shift))
    elif abs(shift) >= 1:
        signal = "* Notable *"
    else:
        signal = ""
    
    if actual > 0 or avg > 0.5:
        print(f"  {key[1]:<33} {avg:>10.1f} {actual:>10} {shift:>+10.1f} {signal}")

print("\n" + "=" * 80)
print("KEY PATTERN CHANGES FOR PAPER 2 PREDICTION:")
print("=" * 80)

for key, avg, actual, shift in big_shifts:
    direction = "INCREASED" if shift > 0 else "DECREASED"
    print(f"\n  [{key[0]}] {key[1]}")
    print(f"    Historical average: {avg:.1f} questions per paper")
    print(f"    2026 Paper 1:      {actual} questions")
    print(f"    Change:            {direction} by {abs(shift):.1f} questions")
    if shift > 0:
        print(f"    >>> EXPECT {actual} questions in Paper 2 as well")
    else:
        print(f"    >>> EXPECT only {actual} questions in Paper 2 as well")

# ── Question format analysis ──
print("\n\n" + "=" * 80)
print("NEW QUESTION FORMATS IN 2026 (not seen in 2022-2025):")
print("=" * 80)

hist_templates = set()
for q in recent_qs:
    words = re.sub(r'[^a-z0-9\s]', '', q.get("question_text", "").lower()).split()
    if len(words) >= 5:
        hist_templates.add(' '.join(words[:5]))

new_templates = []
for q in paper_2026:
    words = re.sub(r'[^a-z0-9\s]', '', q.get("question_text", "").lower()).split()
    if len(words) >= 5:
        prefix = ' '.join(words[:5])
        if prefix not in hist_templates:
            new_templates.append((q["topic"], q["question_text"][:120]))

print(f"\nNew template prefixes not seen in 2022-2025: {len(new_templates)}")
for topic, txt in new_templates[:15]:
    print(f"  [{topic}] {txt}")

# ── Difficulty distribution ──
print("\n\n" + "=" * 80)
print("2026 PAPER 1 DIFFICULTY DISTRIBUTION:")
print("=" * 80)
diff_counts = Counter(q.get("difficulty", "medium") for q in paper_2026)
for diff, count in diff_counts.most_common():
    print(f"  {diff}: {count} questions ({count}%)")
