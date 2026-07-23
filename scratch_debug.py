import json
from pathlib import Path

q_json_path = Path(r'e:\afcatpyq3\data\processed\Q.json')
q_data = json.load(open(q_json_path, encoding='utf-8'))

for q in q_data:
    if q.get('topic') == 'Venn Diagrams' and q.get('file_name') == 'AFCAT_2015_Official_Paper2.pdf':
        print(f"Q{q.get('question_number')}: {str(q.get('question_text'))[:40]}... has_figure={q.get('has_figure')}")
