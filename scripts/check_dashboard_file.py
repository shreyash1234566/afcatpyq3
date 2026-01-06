#!/usr/bin/env python3
from pathlib import Path
p=Path('output/predictions_2026/dashboard.html')
if not p.exists():
    print('File not found',p)
    raise SystemExit(1)
s=p.read_text(encoding='utf-8')
print('Prediction Summary in file:', 'Prediction Summary' in s)
print('Top 10 High Priority Topics in file:', 'Top 10 High Priority Topics' in s)
idx=s.find('Prediction Summary')
if idx!=-1:
    print('\n--- SNIPPET ---\n')
    print(s[idx:idx+800])
