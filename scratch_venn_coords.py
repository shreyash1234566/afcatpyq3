import json
import fitz
from pathlib import Path
from collections import defaultdict
import io
from PIL import Image

BASE = Path(r'e:\afcatpyq3')
Q_JSON_PATH = BASE / 'data/processed/Q.json'
PAPERS_DIR = BASE / 'data/papers'

q_data = json.load(open(Q_JSON_PATH, encoding='utf-8'))

venn_by_paper = defaultdict(list)
for q in q_data:
    if q.get('topic') == 'Venn Diagrams':
        venn_by_paper[q.get('file_name')].append(q)

for paper, qs in venn_by_paper.items():
    qs.sort(key=lambda x: int(x.get('question_number', 0)))
    
    # Blocks
    blocks = []
    current_block = [qs[0]]
    for q in qs[1:]:
        if int(q.get('question_number', 0)) - int(current_block[-1].get('question_number', 0)) <= 2:
            current_block.append(q)
        else:
            blocks.append(current_block)
            current_block = [q]
    blocks.append(current_block)
    
    print(f"\n--- {paper} ---")
    doc = fitz.open(str(PAPERS_DIR / paper))
    
    for b_idx, block in enumerate(blocks):
        q_start = block[0].get('question_number')
        q_text = block[0].get('question_text', '')
        
        # Find page
        page_idx = None
        q_start_y = None
        for i in range(len(doc)):
            page = doc[i]
            # Try finding Q_start
            for block_dict in page.get_text("dict").get("blocks", []):
                if block_dict.get("type") != 0: continue
                for line in block_dict.get("lines", []):
                    for span in line.get("spans", []):
                        t = span["text"].strip()
                        if t.startswith(str(q_start) + ".") or t.startswith("Q" + str(q_start)):
                            page_idx = i
                            q_start_y = line["bbox"][1]
                            break
                    if page_idx is None and q_text:
                        # try fragment
                        words = q_text.split()
                        frag = " ".join(words[:5])
                        if frag in "".join(s["text"] for s in line.get("spans", [])):
                            page_idx = i
                            q_start_y = line["bbox"][1]
                            break
                if page_idx is not None: break
            if page_idx is not None: break
            
        if page_idx is None:
            print(f"  Block Q{q_start}: Could not find on any page!")
            continue
            
        page = doc[page_idx]
        
        # Find Q_next (if inline)
        q_next = q_start + 1
        q_next_y = None
        for block_dict in page.get_text("dict").get("blocks", []):
            if block_dict.get("type") != 0: continue
            for line in block_dict.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if t.startswith(str(q_next) + ".") or t.startswith("Q" + str(q_next)):
                        q_next_y = line["bbox"][1]
                        break
                        
        # Find instruction or Q_prev
        q_prev_y = None
        for block_dict in page.get_text("dict").get("blocks", []):
            if block_dict.get("type") != 0: continue
            for line in block_dict.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if "Directions" in t or "Which of the following diagrams" in t:
                        if line["bbox"][1] < q_start_y:
                            if q_prev_y is None or line["bbox"][1] < q_prev_y:
                                q_prev_y = line["bbox"][1]
        
        print(f"  Block Q{q_start} (Page {page_idx+1}): Q_prev/Inst={q_prev_y}, Q_start={q_start_y}, Q_next={q_next_y}")
