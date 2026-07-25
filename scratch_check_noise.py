import json
import collections

data = json.load(open('e:/afcatpyq3/data/processed/Q.json', encoding='utf-8'))

# Let's find years where "Idioms & Phrases" jumped to 25.
# And "Synonyms/Antonyms" jumped to 41.

for q in data:
    sec = q.get('section')
    y = q.get('file_name', '')
    t = q.get('topic')
    
    # 2024 Verbal Ability anomalies:
    if sec == 'Verbal Ability' and '2024' in y:
        if t in ['Idioms & Phrases', 'Synonyms', 'Antonyms', 'Synonyms/Antonyms']:
            print(f"[{y}] [{t}] {q.get('question_text', '')[:100].strip()}")
