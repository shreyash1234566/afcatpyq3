#!/usr/bin/env python3
"""
Attempt to repair malformed `data/processed/Q.json` by extracting JSON objects.
Backs up the original file to `Q.json.bak` and writes a cleaned list to `Q.json`.
"""
import json
from pathlib import Path
import shutil

P = Path('data/processed/Q.json')
if not P.exists():
    print('Q.json not found at', P)
    raise SystemExit(1)

bak = P.with_suffix('.json.bak')
shutil.copy2(P, bak)
print('Backed up original to', bak)

content = P.read_text(encoding='utf-8')
try:
    data = json.loads(content)
    if isinstance(data, list):
        cleaned = data
    elif isinstance(data, dict) and 'questions' in data:
        cleaned = data['questions']
    else:
        cleaned = [data]
    print('Q.json parsed successfully; writing cleaned list')
    P.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Wrote cleaned Q.json')
    raise SystemExit(0)
except Exception:
    pass

from json import JSONDecoder
decoder = JSONDecoder()
pos = 0
length = len(content)
objs = []
while pos < length:
    try:
        obj, idx = decoder.raw_decode(content, pos)
        pos = idx
        objs.append(obj)
        # skip whitespace
        while pos < length and content[pos].isspace():
            pos += 1
    except Exception:
        pos += 1

# Flatten if objs contain lists
cleaned = []
for o in objs:
    if isinstance(o, list):
        cleaned.extend(o)
    elif isinstance(o, dict):
        # If dict contains 'questions'
        if 'questions' in o and isinstance(o['questions'], list):
            cleaned.extend(o['questions'])
        else:
            cleaned.append(o)

if not cleaned:
    print('Failed to recover any JSON objects from Q.json')
    print('Restoring original from backup')
    shutil.copy2(bak, P)
    raise SystemExit(1)

P.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Recovered {len(cleaned)} items and wrote cleaned Q.json')
