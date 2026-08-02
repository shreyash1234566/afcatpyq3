"""
Generic Figure Extractor via PDF Region Rendering
=================================================
For any question with has_figure=True, this script:
1. Locates Q_n and Q_n+1 Y-coordinates on the page.
2. Renders the region between them as an image.
3. Removes watermarks using HSV color filtering.
4. Crops tight to the remaining non-white content (the figure + options).
5. Saves the image and updates Q.json.
"""

import json
import fitz
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

BASE = Path(r'e:\afcatpyq3')
Q_JSON_PATH = BASE / 'data/processed/Q.json'
PAPERS_DIR = BASE / 'data/papers'
IMAGES_DIR = BASE / 'data/images'


def find_question_positions(doc):
    """Find (page_idx, y) for every question number in the PDF."""
    positions = {}
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                # Q66. or Q66 or 66. or 66)
                m = re.match(r'^Q?\.?\s*(\d+)\s*[.)](?:\s|$)', line_text)
                if m:
                    qnum = int(m.group(1))
                    if qnum not in positions:
                        # Give preference to the top of the block
                        positions[qnum] = (page_idx, line["bbox"][1])
    return positions


def render_region(doc, page_idx, y_top, y_bottom, zoom=2.5):
    """Render a region of a PDF page as a BGR numpy array."""
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_idx]
    clip = fitz.Rect(0, y_top, page.rect.width, y_bottom)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_cross_page(doc, page_idx, y_start, next_page_idx, next_y, zoom=2.5):
    """Render a region that spans two pages, stitched vertically."""
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_idx]
    
    # Bottom of current page
    clip1 = fitz.Rect(0, y_start, page.rect.width, page.rect.height)
    pix1 = page.get_pixmap(matrix=mat, clip=clip1)
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, pix1.n)
    
    # Top of next page
    next_page = doc[next_page_idx]
    clip2 = fitz.Rect(0, 0, next_page.rect.width, next_y)
    pix2 = next_page.get_pixmap(matrix=mat, clip=clip2)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, pix2.n)
    
    # Normalize channels
    if img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)
    else:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    if img2.shape[2] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2BGR)
    else:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    # Match widths
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:, :w]
    img2 = img2[:, :w]
    
    return np.vstack([img1, img2])


def remove_colored_regions(img_bgr):
    """
    Remove colored watermarks, ads, and logos from the image.
    We assume the diagrams are BLACK lines on WHITE background (zero saturation).
    Only remove pixels that are genuinely COLORFUL (high saturation).
    """
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Only mask pixels with HIGH saturation (S > 60) AND reasonable value (V > 50)
    # This catches red/pink watermarks but leaves black lines (S=0) and white bg alone
    mask_color = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([180, 255, 255]))
    
    # Gentle dilation to cover edges of colored regions, but not too much
    kernel = np.ones((5, 5), np.uint8)
    mask_expanded = cv2.dilate(mask_color, kernel, iterations=2)
    
    # Replace colored regions with white
    result[mask_expanded > 0] = [255, 255, 255]
    
    return result


def crop_to_content(img_bgr):
    """
    Crop the image to the bounding box of all non-white pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Threshold to binary (inverted so content is white, bg is black)
    # Use 240 to catch light gray anti-aliasing pixels as well
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    pad = 15
    h_img, w_img = img_bgr.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + 2*pad)
    h = min(h_img - y, h + 2*pad)
    
    cropped = img_bgr[y:y+h, x:x+w]
    
    if cropped.shape[0] < 30 or cropped.shape[1] < 30:
        return None
        
    return cropped


def main():
    q_data = json.load(open(Q_JSON_PATH, encoding='utf-8'))
    
    # We will process ALL questions that have has_figure=True, EXCEPT Venn Diagrams
    # (since we already have a specialized script for Venn diagrams that crops to circles).
    # Wait, actually Venn diagrams are fine to re-extract with this generic method
    # since it will just include the text options too. 
    # But to be safe and preserve the perfect Venn extractions, we'll skip Venn Diagrams if they already have an image.
    
    fig_by_paper = defaultdict(list)
    for q in q_data:
        if q.get('has_figure'):
            if q.get('topic') == 'Venn Diagrams' and len(q.get('image_path', [])) > 0:
                continue # Already perfectly extracted
            fig_by_paper[q.get('file_name')].append(q)
    
    total_ok = 0
    total_fail = 0
    
    for paper, qs in fig_by_paper.items():
        pdf_path = PAPERS_DIR / paper
        if not pdf_path.exists():
            print(f"\n[SKIP] {paper} - PDF not found")
            continue
        
        doc = fitz.open(str(pdf_path))
        pdf_stem = paper.replace(".pdf", "")
        img_dir = IMAGES_DIR / pdf_stem
        img_dir.mkdir(parents=True, exist_ok=True)
        
        all_pos = find_question_positions(doc)
        qs.sort(key=lambda x: int(x.get('question_number', 0)))
        
        print(f"\n{'='*60}")
        print(f"  {paper} ({len(qs)} figure Qs)")
        print(f"{'='*60}")
        
        for q in qs:
            qnum = int(q.get('question_number', 0))
            if qnum == 0:
                continue
                
            q_text = q.get('question_text', '')
            
            # --- Find THIS question's position ---
            page_idx, y_start = None, None
            if qnum in all_pos:
                page_idx, y_start = all_pos[qnum]
            else:
                # Text search fallback
                words = q_text.strip().split()
                for length in range(min(10, len(words)), 3, -1):
                    frag = " ".join(words[:length])
                    for pi in range(len(doc)):
                        rects = doc[pi].search_for(frag)
                        if rects:
                            page_idx, y_start = pi, rects[0].y0
                            break
                    if page_idx is not None:
                        break
            
            if page_idx is None:
                print(f"  Q{qnum}: [FAIL] Not found in PDF")
                q['image_path'] = []
                q['has_figure'] = False
                total_fail += 1
                continue
            
            # --- Find the NEXT question's position (boundary) ---
            next_page_idx, next_y = None, None
            next_qnum = qnum + 1
            if next_qnum in all_pos:
                next_page_idx, next_y = all_pos[next_qnum]
            else:
                # Try +2, +3... up to +5 in case a question is missing/skipped
                for offset in range(2, 6):
                    if (qnum + offset) in all_pos:
                        next_page_idx, next_y = all_pos[qnum + offset]
                        break
            
            # --- Render the question's region ---
            page = doc[page_idx]
            
            # Add a small buffer to y_start so we don't accidentally clip the top of the question
            y_start = max(0, y_start - 10)
            
            if next_page_idx is None:
                # No next question found → crop to page bottom
                img_bgr = render_region(doc, page_idx, y_start, page.rect.height)
            elif next_page_idx == page_idx:
                # Same page → crop between questions
                # Add buffer to next_y to avoid clipping the next question's top
                safe_next_y = max(y_start + 10, next_y - 5)
                img_bgr = render_region(doc, page_idx, y_start, safe_next_y)
            else:
                # Cross page → stitch
                safe_next_y = max(10, next_y - 5)
                img_bgr = render_cross_page(doc, page_idx, y_start, next_page_idx, safe_next_y)
            
            if img_bgr is None or img_bgr.shape[0] < 30:
                print(f"  Q{qnum}: [FAIL] Render too small")
                q['image_path'] = []
                # Keep has_figure = True, maybe we fix later
                total_fail += 1
                continue
            
            # --- Clean and Crop ---
            cleaned = remove_colored_regions(img_bgr)
            cropped = crop_to_content(cleaned)
            
            if cropped is not None:
                out_fname = f"q{qnum}_figure.png"
                out_path = img_dir / out_fname
                cv2.imwrite(str(out_path), cropped)
                
                rel_path = f"data/images/{pdf_stem}/{out_fname}"
                q['image_path'] = [rel_path]
                q['has_figure'] = True
                
                # Simple dark check
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                dark_ratio = np.sum(gray < 40) / gray.size
                q['image_dark'] = bool(dark_ratio > 0.4)
                
                total_ok += 1
                print(f"  Q{qnum}: [OK] {cropped.shape[1]}x{cropped.shape[0]} px")
            else:
                print(f"  Q{qnum}: [FAIL] Cropped to empty")
                q['image_path'] = []
                total_fail += 1
        
        doc.close()
    
    # Save Q.json
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(q_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  DONE: {total_ok} images extracted, {total_fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
