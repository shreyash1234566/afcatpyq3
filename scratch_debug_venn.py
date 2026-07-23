import json
from pathlib import Path
from collections import defaultdict

BASE = Path(r'e:\afcatpyq3')
q_json_path = BASE / 'data/processed/Q.json'
q_data = json.load(open(q_json_path, encoding='utf-8'))

venn_by_paper = defaultdict(list)
for q in q_data:
    if q.get('topic') == 'Venn Diagrams':
        venn_by_paper[q.get('file_name')].append(q)

for paper, qs in venn_by_paper.items():
    qs.sort(key=lambda x: int(x.get('question_number', 0)))
    
    blocks = []
    current_block = [qs[0]]
    for q in qs[1:]:
        if int(q.get('question_number', 0)) - int(current_block[-1].get('question_number', 0)) <= 2:
            current_block.append(q)
        else:
            blocks.append(current_block)
            current_block = [q]
    blocks.append(current_block)
    
    print(f"\nPaper: {paper}")
    for b_idx, block in enumerate(blocks):
        qnums = [q.get('question_number') for q in block]
        all_imgs = set()
        for q in block:
            for img in q.get('image_path', []):
                all_imgs.add(img)
        sorted_imgs = sorted(list(all_imgs))
        
        print(f"  Block {b_idx+1}: Qs {qnums}")
        print(f"    Total Images aggregated: {len(sorted_imgs)}")
        if len(sorted_imgs) == 0:
            print("    [WARNING] NO IMAGES FOUND FOR THIS BLOCK!")
