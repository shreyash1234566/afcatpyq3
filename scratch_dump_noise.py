import json

data = json.load(open('e:/afcatpyq3/data/processed/Q.json', encoding='utf-8'))
out = []
for q in data:
    sec = q.get('section')
    y = q.get('file_name', '')
    t = q.get('topic')
    
    if sec == 'Verbal Ability' and '2024' in y:
        if t in ['Idioms & Phrases', 'Synonyms', 'Antonyms', 'Synonyms/Antonyms']:
            out.append(f"[{y}] [{t}] {q.get('question_text', '')[:150].strip()}")

with open('e:/afcatpyq3/scratch_noise_output_utf8.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(out))
