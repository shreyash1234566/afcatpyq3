#!/usr/bin/env python3
"""
Compute 'certainTopics' from existing aiPredictedQuestions and update data.js
"""
import json
from pathlib import Path

DATA_JS = Path('output/predictions_2026/data.js')
if not DATA_JS.exists():
    print('data.js not found at', DATA_JS)
    raise SystemExit(1)
text = DATA_JS.read_text(encoding='utf-8')
start = text.find('const dashboardData')
if start == -1:
    print('const dashboardData not found')
    raise SystemExit(1)
obj_start = text.find('{', start)
# find matching closing brace
i = obj_start
depth = 0
end_idx = -1
while i < len(text):
    c = text[i]
    if c == '{':
        depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0:
            end_idx = i
            break
    i += 1
if end_idx == -1:
    print('Could not find end of dashboardData object')
    raise SystemExit(1)
json_text = text[obj_start:end_idx+1]
try:
    data = json.loads(json_text)
except Exception as e:
    print('JSON parse error:', e)
    raise

ai_pred = data.get('aiPredictedQuestions', [])
# if aiPredictedQuestions is an object with 'questions' key
if isinstance(ai_pred, dict) and 'questions' in ai_pred:
    ai_list = ai_pred['questions']
elif isinstance(ai_pred, list):
    ai_list = ai_pred
else:
    ai_list = []

# count topics
from collections import Counter
cnt = Counter()
sec_map = {}
for q in ai_list:
    topic = q.get('topic') or q.get('topic_name') or q.get('topic_code')
    section = q.get('section') or q.get('section_name') or q.get('section_code') or 'Unknown'
    if topic:
        cnt[topic] += 1
        sec_map[topic] = section

# choose 'certain' topics: top 5 by count, and any with count >=2
if cnt:
    most_common = [t for t,c in cnt.most_common(10)]
    certain = []
    for t in most_common:
        if cnt[t] >= 2 or len(certain) < 5:
            certain.append({'topic': t, 'section': sec_map.get(t, 'Unknown'), 'count': cnt[t]})
else:
    certain = []

# attach to data
data['certainTopics'] = certain
# rebuild file
header = text[:start]
new_body = 'const dashboardData = ' + json.dumps(data, indent=2, ensure_ascii=False) + ';\\n'
new_text = header + new_body + text[end_idx+2:]
DATA_JS.write_text(new_text, encoding='utf-8')
print('Wrote certainTopics to', DATA_JS)
print('certainTopics:', certain)