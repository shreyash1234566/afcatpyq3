#!/usr/bin/env python3
"""
Fetch the locally served dashboard HTML and verify the Prediction Summary content.
"""
import sys
from urllib.request import urlopen

URL = 'http://localhost:8000/dashboard.html'

try:
    with urlopen(URL, timeout=5) as resp:
        html = resp.read().decode('utf-8', errors='replace')
except Exception as e:
    print('ERROR: Could not fetch', URL, '-', e)
    sys.exit(2)

found_summary = 'Prediction Summary' in html or 'Prediction Summary' in html
found_top10 = 'Top 10 High Priority Topics' in html

print('Fetched:', URL)
print('Prediction Summary block found:', found_summary)
print('Top 10 High Priority Topics found:', found_top10)

if found_summary:
    # print a short snippet
    idx = html.find('Prediction Summary')
    snippet = html[idx: idx+800] if idx!=-1 else html[:800]
    print('\n--- SNIPPET ---\n')
    print(snippet)
