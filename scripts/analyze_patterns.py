import json, re
from collections import defaultdict

data = json.load(open('data/processed/Q.json', encoding='utf-8'))

def get_year(fn):
    for yr in range(2011, 2025):
        if str(yr) in fn:
            return yr
    return None

# ────────────────────────────────────────────────
# 1. Structural template analysis
# ────────────────────────────────────────────────
starters = defaultdict(list)
for q in data:
    txt = q.get('question_text', '').strip()
    words = txt.split()
    if len(words) >= 4:
        key = ' '.join(words[:5]).lower()
        yr = get_year(q.get('file_name', ''))
        starters[key].append({'year': yr, 'text': txt[:120], 'topic': q.get('topic')})

# Templates that appear in 4+ different years
popular = [(k, v) for k, v in starters.items() if len(set(x['year'] for x in v if x['year'])) >= 4]
popular.sort(key=lambda x: len(set(y['year'] for y in x[1] if y['year'])), reverse=True)
print(f'Template patterns used in 4+ different years: {len(popular)}')
for k, v in popular[:15]:
    yrs = sorted(set(y['year'] for y in v if y['year']))
    print(f'  {len(yrs)} years {yrs}: "{k}..."')
    print(f'    Topic: {v[0]["topic"]}')

print()

# ────────────────────────────────────────────────
# 2. Year-over-year topic repetition
# ────────────────────────────────────────────────
topic_by_year = defaultdict(lambda: defaultdict(int))
for q in data:
    yr = get_year(q.get('file_name', ''))
    if yr:
        topic_by_year[yr][q.get('topic', '')] += 1

print("\nTop topics per year (GK section):")
for yr in sorted(topic_by_year.keys()):
    top = sorted(topic_by_year[yr].items(), key=lambda x: -x[1])[:3]
    print(f"  {yr}: {top}")

# ────────────────────────────────────────────────
# 3. Exact question recycling rate
# ────────────────────────────────────────────────
text_years = defaultdict(list)
for q in data:
    txt = q.get('question_text', '').strip().lower()
    yr = get_year(q.get('file_name', ''))
    if yr and txt:
        text_years[txt].append(yr)

recycled = [(t, sorted(set(y))) for t, y in text_years.items() if len(set(y)) > 1]
print(f"\nTotal questions recycled across years: {len(recycled)}")
recycled.sort(key=lambda x: len(x[1]), reverse=True)
for txt, yrs in recycled[:5]:
    print(f"  Appeared in {yrs}: {txt[:80]}...")
