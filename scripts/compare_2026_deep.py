"""
Compare the ACTUAL 2026 vocabulary words against our historical word bank.
Check if any 2026 words were previously asked.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
history = json.loads((ROOT / "data" / "processed" / "Q_clean.json").read_text(encoding="utf-8"))
actual_2026 = json.loads((ROOT / "data" / "papers" / "afcat_2026_questions.json").read_text(encoding="utf-8"))

def get_year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

# Build historical word bank
hist_words = defaultdict(list)  # word -> [(year, question)]
for q in history:
    if q.get("topic") not in ["Synonyms/Antonyms"]: continue
    txt = q.get("question_text", "")
    yr = get_year(q.get("file_name", ""))
    caps = re.findall(r'\b([A-Z]{4,})\b', txt)
    skip = {"MOST", "WORD", "GIVEN", "FOLLOWING", "SELECT", "CHOOSE", "OPTION", 
            "SENTENCE", "BEST", "MEANING", "NEAREST", "APPROPRIATE", "UNDERLINED",
            "BELOW", "WHICH", "FIND", "CLOSEST", "THAT", "WITH", "FROM", "EACH",
            "AFCAT", "QUESTION", "ANSWER", "CORRECT", "ONE", "THE", "OPPOSITE",
            "NEARLY", "EXPRESSES", "ANTONYM", "SYNONYM", "ALTERNATIVES"}
    for w in caps:
        if w not in skip and len(w) >= 4:
            hist_words[w].append((yr, txt[:80]))

# Extract 2026 vocabulary words
print("=" * 70)
print("2026 VOCAB WORDS vs HISTORICAL WORD BANK")
print("=" * 70)

for q in actual_2026:
    if q.get("topic") not in ["Synonyms/Antonyms"]: continue
    txt = q.get("question_text", "")
    caps = re.findall(r'\b([A-Z]{4,})\b', txt)
    skip = {"MOST", "WORD", "GIVEN", "FOLLOWING", "SELECT", "CHOOSE", "OPTION", 
            "SENTENCE", "BEST", "MEANING", "NEAREST", "APPROPRIATE", "UNDERLINED",
            "BELOW", "WHICH", "FIND", "CLOSEST", "THAT", "WITH", "FROM", "EACH",
            "AFCAT", "QUESTION", "ANSWER", "CORRECT", "ONE", "THE", "OPPOSITE",
            "NEARLY", "EXPRESSES", "ANTONYM", "SYNONYM", "ALTERNATIVES",
            "NONE", "INTERPRETATION", "GOVERNANCE", "CONSTITUTION"}
    for w in caps:
        if w not in skip and len(w) >= 4:
            if w in hist_words:
                years_seen = [y for y, _ in hist_words[w]]
                print(f"  [REPEAT!] '{w}' -> asked in 2026 AND previously in {sorted(set(years_seen))}")
            else:
                print(f"  [NEW]     '{w}' -> asked in 2026 for the FIRST TIME")

# Also check idioms
print("\n" + "=" * 70)
print("2026 IDIOM COMPARISON")
print("=" * 70)

hist_idioms = set()
for q in history:
    if q.get("topic") not in ["Idioms & Phrases"]: continue
    txt = q.get("question_text", "").lower()
    m = re.search(r'(?:idiom|phrase)[:\s]*(.+?)(?:\n|$)', txt)
    if m:
        hist_idioms.add(m.group(1).strip()[:50])

for q in actual_2026:
    if q.get("topic") not in ["Idioms & Phrases"]: continue
    txt = q.get("question_text", "").lower()
    m = re.search(r'(?:idiom|phrase)[:\s]*(.+?)(?:\n|$)', txt)
    if m:
        idiom_2026 = m.group(1).strip()[:50]
        if idiom_2026 in hist_idioms:
            print(f"  [REPEAT!] '{idiom_2026}'")
        else:
            print(f"  [NEW]     '{idiom_2026}'")

# GK topic distribution comparison
print("\n" + "=" * 70)
print("2026 TOPIC COUNT vs DM PREDICTION")
print("=" * 70)

actual_topics = defaultdict(int)
for q in actual_2026:
    actual_topics[(q["section"], q["topic"])] += 1

# Load generated
generated = json.loads((ROOT / "output" / "generated_questions" / "practice_2026_sota.json").read_text(encoding="utf-8"))
pred_topics = defaultdict(int)
for q in generated:
    pred_topics[(q["section"], q["topic"])] += 1

print(f"\n{'Topic':<35} {'Actual':>8} {'Predicted':>10} {'Match?':>8}")
print("-" * 65)

all_keys = sorted(set(list(actual_topics.keys()) + list(pred_topics.keys())))
exact_matches = 0
close_matches = 0
for key in all_keys:
    actual = actual_topics.get(key, 0)
    predicted = pred_topics.get(key, 0)
    diff = abs(actual - predicted)
    match = "EXACT" if diff == 0 else ("CLOSE" if diff <= 1 else f"OFF by {diff}")
    if diff == 0: exact_matches += 1
    if diff <= 1: close_matches += 1
    print(f"  {key[1]:<33} {actual:>8} {predicted:>10} {match:>8}")

total = len(all_keys)
print(f"\n  EXACT matches: {exact_matches}/{total} ({exact_matches/total*100:.0f}%)")
print(f"  CLOSE matches (within 1): {close_matches}/{total} ({close_matches/total*100:.0f}%)")
