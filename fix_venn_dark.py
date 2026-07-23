import json
from pathlib import Path

BASE = Path(r'e:\afcatpyq3')
q_json_path = BASE / 'data/processed/Q.json'
data_js_path = BASE / 'output/predictions_2026/data.js'

print("Fixing Q.json image_dark for Venn Diagrams...")
data = json.load(open(q_json_path, encoding='utf-8'))
fixed = 0
for q in data:
    if q.get('topic') == 'Venn Diagrams' and q.get('image_dark'):
        q['image_dark'] = False
        fixed += 1
with open(q_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Fixed {fixed} questions in Q.json.")

print("Fixing data.js image_dark for Venn Diagrams...")
with open(data_js_path, 'r', encoding='utf-8') as f:
    content = f.read().strip()
    prefix = 'const dashboardData = '
    if content.startswith(prefix):
        content = content[len(prefix):]
    if content.endswith(';'):
        content = content[:-1]

dash_data = json.loads(content)
fixed_js = 0

for q in dash_data.get('question_bank', []):
    if q.get('topic') == 'Venn Diagrams' and q.get('image_dark'):
        q['image_dark'] = False
        fixed_js += 1

for q in dash_data.get('mock_test', {}).get('all_questions', []):
    if q.get('topic') == 'Venn Diagrams' and q.get('image_dark'):
        q['image_dark'] = False

new_content = prefix + json.dumps(dash_data, separators=(',', ':'), ensure_ascii=False) + ";\n"
with open(data_js_path, "w", encoding="utf-8") as f:
    f.write(new_content)
print(f"Fixed {fixed_js} questions in data.js.")
