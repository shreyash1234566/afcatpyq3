import json
from pathlib import Path

p = Path('output/predictions_2026/data.js')
if not p.exists():
    print('data.js not found at', p)
    raise SystemExit(1)
text = p.read_text(encoding='utf-8')
start = text.find('const dashboardData')
if start == -1:
    print('const dashboardData not found')
    raise SystemExit(1)
# find first '{' after start
obj_start = text.find('{', start)
# find matching closing '};' by counting braces
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
# load JSON
try:
    data = json.loads(json_text)
except Exception as e:
    print('JSON parse error:', e)
    raise

print('dashboardData keys:', list(data.keys()))
ai_pred = data.get('aiPredictedQuestions')
if ai_pred is None:
    print('aiPredictedQuestions key missing')
else:
    print('aiPredictedQuestions length:', len(ai_pred))
    topics = []
    for q in ai_pred:
        topic = q.get('topic') or q.get('topic_name') or q.get('topic_code')
        section = q.get('section') or q.get('section_name') or q.get('section_code')
        topics.append((topic, section))
    uniq = []
    for t,s in topics:
        if not any(u[0]==t and u[1]==s for u in uniq):
            uniq.append((t,s))
    print('Unique predicted topics count:', len(uniq))
    for i,(t,s) in enumerate(uniq[:30],1):
        print(i, t, '->', s)
