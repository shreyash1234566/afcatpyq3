"""
Deep pattern analysis of AFCAT past papers to find:
1. Repeated vocabulary words across years
2. Grammar error types that recur
3. GK topic cycling patterns
4. Math formula recycling
5. Idiom/phrase reuse
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "data" / "processed" / "Q_clean.json").read_text(encoding="utf-8"))

def get_year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

# ============================================================
# 1. VOCABULARY WORD EXTRACTION (Synonyms/Antonyms)
# ============================================================
print("=" * 70)
print("1. VOCABULARY WORDS ASKED IN AFCAT (Synonyms/Antonyms)")
print("=" * 70)

vocab_qs = [q for q in data if q.get("topic") in ["Synonyms/Antonyms"]]
word_years = defaultdict(list)  # word -> [years it appeared]

for q in vocab_qs:
    txt = q.get("question_text", "").upper()
    yr = get_year(q.get("file_name", ""))
    # Extract the target word (usually in CAPS, quotes, or after "word")
    # Pattern 1: "synonym of the word 'XYZ'"
    m = re.search(r"(?:synonym|antonym)\s+(?:of\s+)?(?:the\s+)?(?:word\s+)?['\"]?([A-Z]{3,})['\"]?", txt, re.IGNORECASE)
    if m:
        word_years[m.group(1).upper()].append(yr)
        continue
    # Pattern 2: word in ALL CAPS
    caps = re.findall(r'\b([A-Z]{4,})\b', txt)
    # Filter out common words
    skip = {"MOST", "WORD", "GIVEN", "FOLLOWING", "SELECT", "CHOOSE", "OPTION", 
            "SENTENCE", "BEST", "MEANING", "NEAREST", "APPROPRIATE", "UNDERLINED",
            "BELOW", "WHICH", "FIND", "CLOSEST", "THAT", "WITH", "FROM", "EACH",
            "AFCAT", "QUESTION", "ANSWER", "CORRECT", "ONE", "THE"}
    for w in caps:
        if w not in skip and len(w) >= 4:
            word_years[w].append(yr)

# Find repeated words
repeated = {w: yrs for w, yrs in word_years.items() if len(yrs) >= 2}
print(f"\nTotal unique vocabulary words extracted: {len(word_years)}")
print(f"Words that appeared in 2+ different papers: {len(repeated)}")
print("\nREPEATED WORDS:")
for w, yrs in sorted(repeated.items(), key=lambda x: -len(x[1])):
    print(f"  {w}: appeared {len(yrs)} times in years {sorted(set(yrs))}")

# Most recent words (2023-2025)
recent_words = [w for w, yrs in word_years.items() if any(y and y >= 2023 for y in yrs)]
print(f"\nWords from 2023-2025 papers ({len(recent_words)} words):")
for w in sorted(recent_words)[:30]:
    print(f"  {w}")

# ============================================================
# 2. IDIOMS & PHRASES ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("2. IDIOMS & PHRASES PATTERN")
print("=" * 70)

idiom_qs = [q for q in data if q.get("topic") in ["Idioms & Phrases"]]
idiom_texts = defaultdict(list)
for q in idiom_qs:
    txt = q.get("question_text", "")
    yr = get_year(q.get("file_name", ""))
    # Extract the idiom (usually in quotes or bold)
    m = re.search(r'["\'](.+?)["\']', txt)
    if m:
        idiom_texts[m.group(1).lower().strip()].append(yr)
    else:
        # Try to extract after common patterns
        m2 = re.search(r'(?:idiom|phrase|meaning of)\s*[:\-]?\s*(.+?)(?:\n|$)', txt, re.IGNORECASE)
        if m2:
            idiom_texts[m2.group(1).strip().lower()[:60]].append(yr)

repeated_idioms = {i: yrs for i, yrs in idiom_texts.items() if len(yrs) >= 2}
print(f"\nTotal unique idioms extracted: {len(idiom_texts)}")
print(f"Idioms repeated across papers: {len(repeated_idioms)}")
for idiom, yrs in sorted(repeated_idioms.items(), key=lambda x: -len(x[1]))[:15]:
    print(f"  '{idiom}': {len(yrs)} times in {sorted(set(yrs))}")

# ============================================================
# 3. SPOTTING ERRORS - GRAMMAR ERROR TYPES
# ============================================================
print("\n" + "=" * 70)
print("3. SPOTTING ERRORS - GRAMMAR ERROR TYPES")
print("=" * 70)

error_qs = [q for q in data if q.get("topic") in ["Spotting Errors"]]
error_keywords = Counter()
for q in error_qs:
    txt = q.get("question_text", "").lower()
    if "subject" in txt and "verb" in txt: error_keywords["Subject-Verb Agreement"] += 1
    if "preposition" in txt: error_keywords["Preposition Error"] += 1
    if "tense" in txt: error_keywords["Tense Error"] += 1
    if "article" in txt or " a " in txt or " an " in txt or " the " in txt: error_keywords["Article Usage"] += 1
    if "no error" in txt: error_keywords["Has 'No Error' option"] += 1
    # Check actual error in options/explanation
    expl = q.get("explanation", "").lower()
    if "plural" in expl or "singular" in expl: error_keywords["Singular/Plural Error"] += 1
    if "preposition" in expl: error_keywords["Preposition Error (from explanation)"] += 1
    if "tense" in expl: error_keywords["Tense Error (from explanation)"] += 1
    if "article" in expl: error_keywords["Article Error (from explanation)"] += 1

print(f"\nTotal Spotting Errors questions: {len(error_qs)}")
for etype, count in error_keywords.most_common():
    print(f"  {etype}: {count} times")

# ============================================================
# 4. MATH - FORMULA REPETITION
# ============================================================
print("\n" + "=" * 70)
print("4. NUMERICAL ABILITY - CONCEPT PATTERNS")
print("=" * 70)

math_topics = ["Time & Work", "Time, Speed & Dist.", "Profit & Loss", 
               "Simple/Compound Int.", "Ratio & Proportion", "Average",
               "Percentage", "Mensuration", "Algebra"]

for mt in math_topics:
    qs = [q for q in data if q.get("topic") == mt]
    yr_counts = Counter(get_year(q.get("file_name", "")) for q in qs)
    print(f"\n  [{mt}] Total: {len(qs)}")
    print(f"    Year spread: {dict(sorted(yr_counts.items()))}")

# ============================================================
# 5. GK - REPEATED FACTS
# ============================================================
print("\n" + "=" * 70)
print("5. GENERAL AWARENESS - QUESTION REPETITION")
print("=" * 70)

gk_qs = [q for q in data if q.get("section") == "General Awareness"]
# Check for near-duplicate questions
from itertools import combinations

gk_texts = [(i, q.get("question_text", "").lower()[:80], get_year(q.get("file_name", ""))) 
            for i, q in enumerate(gk_qs)]

# Simple duplicate check
seen_prefixes = defaultdict(list)
for idx, txt, yr in gk_texts:
    prefix = re.sub(r'[^a-z0-9 ]', '', txt)[:50]
    seen_prefixes[prefix].append((yr, txt))

repeated_gk = {p: entries for p, entries in seen_prefixes.items() if len(entries) >= 2 and len(p) > 15}
print(f"\nGK questions with near-duplicate text across years: {len(repeated_gk)}")
for prefix, entries in sorted(repeated_gk.items(), key=lambda x: -len(x[1]))[:15]:
    years = [e[0] for e in entries]
    print(f"  '{prefix[:50]}...' -> appeared in {sorted(set(y for y in years if y))}")

# ============================================================
# 6. ONE WORD SUBSTITUTION - REPEATED CONCEPTS
# ============================================================
print("\n" + "=" * 70)
print("6. ONE WORD SUBSTITUTION - CONCEPTS")
print("=" * 70)

ows_qs = [q for q in data if q.get("topic") == "One Word Substitution"]
ows_answers = defaultdict(list)
for q in ows_qs:
    ans = q.get("correct_answer", "")
    yr = get_year(q.get("file_name", ""))
    if ans and len(ans) > 2:
        ows_answers[ans.lower()].append(yr)

repeated_ows = {a: yrs for a, yrs in ows_answers.items() if len(yrs) >= 2}
print(f"\nTotal OWS questions: {len(ows_qs)}")
print(f"Answers repeated across years: {len(repeated_ows)}")
for ans, yrs in sorted(repeated_ows.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  '{ans}': {len(yrs)} times in {sorted(set(yrs))}")
