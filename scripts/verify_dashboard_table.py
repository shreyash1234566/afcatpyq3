#!/usr/bin/env python3
"""
Verify dashboard Predicted Topics table by reading `output/predictions_2026/data.js` and
printing the `certainTopics` entries as table rows.
"""
import json
from pathlib import Path
import sys

DATA_JS = Path('output/predictions_2026/data.js')
if not DATA_JS.exists():
    print('data.js not found at', DATA_JS)
    sys.exit(1)

text = DATA_JS.read_text(encoding='utf-8')
start = text.find('const dashboardData')
if start == -1:
    print('const dashboardData not found in data.js')
    sys.exit(1)

obj_start = text.find('{', start)
if obj_start == -1:
    print('Could not find JSON object start')
    sys.exit(1)

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
    sys.exit(1)

json_text = text[obj_start:end_idx+1]
try:
    data = json.loads(json_text)
except Exception as e:
    print('JSON parse error:', e)
    sys.exit(1)

certain = data.get('certainTopics') or []
if not certain:
    print('No certainTopics found in dashboardData')
    sys.exit(0)

print('Predicted Topics Likely to Appear (from data.js):')
print(f"{'Topic':40} | {'Section':15} | {'Count':5}")
print('-' * 66)
for t in certain:
    topic = t.get('topic', '')
    section = t.get('section', '')
    count = t.get('count', '')
    print(f"{topic:40} | {section:15} | {count:5}")
