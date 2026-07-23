import re
from pathlib import Path

html_path = Path(r'e:\afcatpyq3\output\predictions_2026\index.html')
html = html_path.read_text(encoding='utf-8')

# Find the backslash-escaped backticks and remove the backslash
new_html = html.replace('\\`<img', '`<img').replace('/>\\`', '/>`')

html_path.write_text(new_html, encoding='utf-8')
print("index.html fixed!")
