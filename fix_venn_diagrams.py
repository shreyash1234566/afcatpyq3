import json
import fitz
from pathlib import Path
from collections import defaultdict
import io
from PIL import Image

BASE = Path(r'e:\afcatpyq3')
Q_JSON_PATH = BASE / 'data/processed/Q.json'
PAPERS_DIR = BASE / 'data/papers'
IMAGES_DIR = BASE / 'data/images'

WATERMARK_MAX_Y = 100.0

def is_watermark_or_logo(rect, img_bytes):
    if rect.y0 < WATERMARK_MAX_Y:
        return True
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pixels = list(img.getdata())
        step = max(1, len(pixels) // 200)
        sample = pixels[::step][:200]
        total = len(sample)
        
        dark = sum(1 for r,g,b in sample if r<40 and g<40 and b<40)
        pink = sum(1 for r,g,b in sample if r>180 and g>120 and b>120 and r>g and r>b and abs(g-b)<50)
        red = sum(1 for r,g,b in sample if r>140 and r>g*1.4 and r>b*1.4)
        
        dark_ratio = dark / total
        pink_ratio = pink / total
        red_ratio = red / total
        
        if dark_ratio > 0.5 and pink_ratio > 0.04:
            return True
        if red_ratio > 0.18:
            return True
    except:
        pass
    return False

def get_page_for_question(doc, q_text, qnum):
    # Try text search
    words = q_text.strip().split()
    for length in range(min(12, len(words)), 4, -1):
        fragment = " ".join(words[:length])
        for page_idx in range(len(doc)):
            if doc[page_idx].search_for(fragment):
                return page_idx
    # Try question number
    import re
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

    total_images_extracted = 0
    total_blocks_updated = 0
    
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
        
        for b_idx, block in enumerate(blocks):
            all_imgs = set()
            for q in block:
                for img in q.get('image_path', []):
                    all_imgs.add(img)
            sorted_imgs = sorted(list(all_imgs))
            
            if len(sorted_imgs) > 0:
                # Assign to all
                for q in block:
                    q['image_path'] = sorted_imgs
                    q['has_figure'] = True
                    q['image_dark'] = False  # Always force Venn Diagrams to NOT invert
                total_blocks_updated += 1
            else:
                print(f"[{paper}] Block {b_idx+1} (Qs {[q['question_number'] for q in block]}) has 0 images! Extracting from PDF...")
                pdf_path = PAPERS_DIR / paper
                if not pdf_path.exists():
                    print("  PDF not found!")
                    continue
                
                doc = fitz.open(str(pdf_path))
                q_text = block[0].get('question_text', '')
                qnum = block[0].get('question_number')
                page_idx = get_page_for_question(doc, q_text, qnum)
                
                if page_idx is None:
                    print("  Could not find page for question!")
                    doc.close()
                    continue
                
                print(f"  Found on page {page_idx+1}")
                page = doc[page_idx]
                
                pdf_stem = paper.replace(".pdf", "")
                img_dir = IMAGES_DIR / pdf_stem
                img_dir.mkdir(parents=True, exist_ok=True)
                
                extracted_paths = []
                seen_xrefs = set()
                
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs: continue
                    seen_xrefs.add(xref)
                    
                    try:
                        rects = page.get_image_rects(xref)
                        if not rects: continue
                        rect = rects[0]
                        
                        base_image = doc.extract_image(xref)
                        if not base_image: continue
                        
                        w, h = base_image["width"], base_image["height"]
                        if w < 40 or h < 40: continue
                        
                        img_bytes = base_image["image"]
                        ext = base_image["ext"]
                        
                        if is_watermark_or_logo(rect, img_bytes):
                            continue
                        
                        fname = f"p{page_idx+1}_venn_{xref}.{ext}"
                        fpath = img_dir / fname
                        with open(fpath, "wb") as f:
                            f.write(img_bytes)
                            
                        rel = str(fpath.relative_to(BASE)).replace("\\", "/")
                        extracted_paths.append(rel)
                    except:
                        pass
                        
                print(f"  Extracted {len(extracted_paths)} images!")
                total_images_extracted += len(extracted_paths)
                
                # Assign to block
                for q in block:
                    q['image_path'] = extracted_paths
                    q['has_figure'] = True
                    q['image_dark'] = False # Always force Venn Diagrams to NOT invert
                
                total_blocks_updated += 1
                doc.close()

    # Save Q.json
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(q_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Extracted {total_images_extracted} new images.")
    print(f"Updated {total_blocks_updated} Venn Diagram blocks.")

if __name__ == "__main__":
    main()
