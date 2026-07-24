import json
import fitz
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(r'e:\afcatpyq3')
Q_JSON_PATH = BASE / 'data/processed/Q.json'
PAPERS_DIR = BASE / 'data/papers'
IMAGES_DIR = BASE / 'data/images'

def extract_venn_by_circles(pdf_path, page_idx, out_path):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    
    # Render at 200 DPI for good resolution
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to OpenCV format (BGR)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get black shapes
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter for circle-like objects (aspect ratio ~1, decent size)
        if 20 < w < 400 and 20 < h < 400:
            aspect = w / float(h)
            if 0.5 < aspect < 2.0:
                # Check solidity/area
                area = cv2.contourArea(cnt)
                if area > 50:
                    circle_boxes.append((x, y, x+w, y+h))
                    
    if not circle_boxes:
        # Try again with a lower threshold if lines are faint
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 20 < w < 400 and 20 < h < 400:
                aspect = w / float(h)
                if 0.5 < aspect < 2.0 and cv2.contourArea(cnt) > 50:
                    circle_boxes.append((x, y, x+w, y+h))
                    
    if not circle_boxes:
        return False
        
    # Get bounding box of ALL circles
    min_x = min(b[0] for b in circle_boxes)
    min_y = min(b[1] for b in circle_boxes)
    max_x = max(b[2] for b in circle_boxes)
    max_y = max(b[3] for b in circle_boxes)
    
    # Add padding
    pad = 40
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(img_array.shape[1], max_x + pad)
    max_y = min(img_array.shape[0], max_y + pad)
    
    # In some papers (e.g. 2015), the circles are in the left column and questions on the right.
    # The padding might capture some text. That's fine.
    
    cropped = img_array[min_y:max_y, min_x:max_x]
    cv2.imwrite(str(out_path), cropped)
    return True

def get_page_for_question(doc, q_text, qnum):
    import re
    # Try text search
    words = q_text.strip().split()
    for length in range(min(12, len(words)), 4, -1):
        fragment = " ".join(words[:length])
        for page_idx in range(len(doc)):
            if doc[page_idx].search_for(fragment):
                return page_idx
    # Try question number
    patterns = [rf"^{qnum}[.)]\s*", rf"^Q\.?\s*{qnum}\b", rf"^\({qnum}\)"]
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans: continue
                line_text = "".join(s["text"] for s in spans).strip()
                for pat in patterns:
                    if re.match(pat, line_text):
                        return page_idx
    return None

def main():
    q_data = json.load(open(Q_JSON_PATH, encoding='utf-8'))
    
    venn_by_paper = defaultdict(list)
    for q in q_data:
        if q.get('topic') == 'Venn Diagrams':
            venn_by_paper[q.get('file_name')].append(q)

    total_extracted = 0
    
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
        
        pdf_path = PAPERS_DIR / paper
        if not pdf_path.exists():
            continue
            
        doc = fitz.open(str(pdf_path))
        pdf_stem = paper.replace(".pdf", "")
        img_dir = IMAGES_DIR / pdf_stem
        img_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {paper}...")
        
        for b_idx, block in enumerate(blocks):
            q_text = block[0].get('question_text', '')
            qnum = block[0].get('question_number')
            
            page_idx = get_page_for_question(doc, q_text, qnum)
            
            if page_idx is None:
                print(f"  [!] Block {b_idx+1} (Q{qnum}): Could not find page.")
                # Clear images just in case they were duplicates
                for q in block:
                    q['image_path'] = []
                continue
                
            out_fname = f"perfect_venn_p{page_idx+1}_b{b_idx+1}.png"
            out_path = img_dir / out_fname
            
            success = extract_venn_by_circles(str(pdf_path), page_idx, out_path)
            
            if success:
                print(f"  [OK] Block {b_idx+1} (Q{qnum}): Extracted unified CV crop.")
                rel_path = f"data/images/{pdf_stem}/{out_fname}"
                for q in block:
                    q['image_path'] = [rel_path]
                    q['has_figure'] = True
                    q['image_dark'] = False
                total_extracted += 1
            else:
                print(f"  [-] Block {b_idx+1} (Q{qnum}): No circles found on page {page_idx+1}.")
                # Leave existing images or clear?
                # The existing images are likely broken duplicates. It's better to clear them
                # if there are actually no figures on this page (memory paper).
                for q in block:
                    q['image_path'] = []
                    q['has_figure'] = False
                    
        doc.close()

    # Save Q.json
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(q_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Perfectly extracted {total_extracted} Venn blocks using Computer Vision.")

if __name__ == "__main__":
    main()
