#!/usr/bin/env python3
"""
Fetch the locally served dashboard and print the Predicted Topics table rows.
"""
import sys
from urllib.request import urlopen
import re

URL = 'http://localhost:8000/dashboard.html'

try:
    with urlopen(URL, timeout=5) as resp:
        html = resp.read().decode('utf-8', errors='replace')
except Exception as e:
    print('ERROR: Could not fetch', URL, '-', e)
    sys.exit(2)

m = re.search(r'<tbody[^>]*id=["\']predicted-topics-table["\'][^>]*>([\s\S]*?)</tbody>', html, re.I)
if not m:
    print('Predicted topics table not found in HTML')
    sys.exit(0)

tbody = m.group(1)
rows = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', tbody, re.I)
if not rows:
    print('No rows found in predicted topics table')
    sys.exit(0)

print('Predicted Topics Table Rows:')
for i, r in enumerate(rows, 1):
    # extract td text
    cols = re.findall(r'<td[^>]*>([\s\S]*?)</td>', r, re.I)
    cols = [re.sub(r'<[^>]+>', '', c).strip() for c in cols]
    print(i, '|'.join(cols))
