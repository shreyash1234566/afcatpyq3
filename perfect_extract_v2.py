"""
Text-to-Text (Text-to-Ext) Exact Figure Extractor
=================================================
This script correctly extracts images from PDFs by ignoring fragile Q-numbers 
and searching the PDF directly for the exact text of the question from Q.json.
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


def clean_text_for_search(text):
    """Normalize text to make it easy to find in a PDF."""
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove weird characters that might be mis-OCR'd
    text = re.sub(r'[^\w\s-]', '', text)
    return text


def find_y_by_text(doc, q_text):
    """Find the y-coordinate by searching the exact text in the PDF."""
    if not q_text or len(q_text) < 5:
        return None, None
        
    cleaned = clean_text_for_search(q_text)
    words = cleaned.split()
    
    # Try searching with different lengths of the prefix
    for length in range(min(15, len(words)), 3, -1):
        frag = " ".join(words[:length])
        for pi in range(len(doc)):
            rects = doc[pi].search_for(frag)
            if rects:
                # Return page_index and the top y-coordinate
                return pi, rects[0].y0
                
    return None, None


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
    
    clip1 = fitz.Rect(0, y_start, page.rect.width, page.rect.height)
    pix1 = page.get_pixmap(matrix=mat, clip=clip1)
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, pix1.n)
    
    next_page = doc[next_page_idx]
    clip2 = fitz.Rect(0, 0, next_page.rect.width, next_y)
    pix2 = next_page.get_pixmap(matrix=mat, clip=clip2)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, pix2.n)
    
    if img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)
    else:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    if img2.shape[2] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2BGR)
    else:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:, :w]
    img2 = img2[:, :w]
    return np.vstack([img1, img2])


def remove_colored_regions(img_bgr):
    """Remove colored watermarks."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([180, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask_expanded = cv2.dilate(mask_color, kernel, iterations=2)
    result = img_bgr.copy()
    result[mask_expanded > 0] = [255, 255, 255]
    return result


def crop_to_content(img_bgr):
    """Crop the image tightly to the content."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
    
    # Group ALL questions by file_name so we know their order
    qs_by_paper = defaultdict(list)
    for idx, q in enumerate(q_data):
        qs_by_paper[q.get('file_name')].append((idx, q))
    
    total_ok = 0
    total_fail = 0
    
    for paper, qs in qs_by_paper.items():
        if not paper: continue
        pdf_path = PAPERS_DIR / paper
        if not pdf_path.exists():
            continue
            
        # Only process paper if it has at least one has_figure=True question
        needs_processing = any(q.get('has_figure') for _, q in qs)
        if not needs_processing:
            continue
            
        doc = fitz.open(str(pdf_path))
        pdf_stem = paper.replace(".pdf", "")
        img_dir = IMAGES_DIR / pdf_stem
        img_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}\n  {paper}\n{'='*60}")
        
        for i in range(len(qs)):
            global_idx, q = qs[i]
            if not q.get('has_figure'):
                continue
                
            qnum = q.get('question_number', 'X')
            
            # --- Text to Ext Logic ---
            page_idx, y_start = find_y_by_text(doc, q.get('question_text', ''))
            
            if page_idx is None:
                print(f"  Q{qnum}: [FAIL] Text not found in PDF")
                q_data[global_idx]['image_path'] = []
                total_fail += 1
                continue
                
            # Find NEXT question's boundary
            next_page_idx, next_y = None, None
            for j in range(i+1, min(i+4, len(qs))):
                nxt_q = qs[j][1]
                p, y = find_y_by_text(doc, nxt_q.get('question_text', ''))
                if p is not None:
                    next_page_idx, next_y = p, y
                    break
                    
            page = doc[page_idx]
            y_start = max(0, y_start - 15) # Small buffer above text
            
            if next_page_idx is None:
                # Crop to page bottom
                img_bgr = render_region(doc, page_idx, y_start, page.rect.height)
            elif next_page_idx == page_idx:
                safe_next_y = max(y_start + 10, next_y - 10)
                img_bgr = render_region(doc, page_idx, y_start, safe_next_y)
            else:
                safe_next_y = max(10, next_y - 10)
                img_bgr = render_cross_page(doc, page_idx, y_start, next_page_idx, safe_next_y)
                
            if img_bgr is None or img_bgr.shape[0] < 30:
                print(f"  Q{qnum}: [FAIL] Render too small")
                q_data[global_idx]['image_path'] = []
                total_fail += 1
                continue
                
            cleaned = remove_colored_regions(img_bgr)
            cropped = crop_to_content(cleaned)
            
            if cropped is not None:
                out_fname = f"q{qnum}_figure.png"
                out_path = img_dir / out_fname
                cv2.imwrite(str(out_path), cropped)
                
                rel_path = f"data/images/{pdf_stem}/{out_fname}"
                q_data[global_idx]['image_path'] = [rel_path]
                q_data[global_idx]['has_figure'] = True
                total_ok += 1
                print(f"  Q{qnum}: [OK] Extracted exact Text-to-Text boundary")
            else:
                q_data[global_idx]['image_path'] = []
                total_fail += 1
                print(f"  Q{qnum}: [FAIL] Cropped to empty")
                
        doc.close()
        
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(q_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nDONE: {total_ok} extracted flawlessly via Text-to-Ext, {total_fail} failed")

if __name__ == "__main__":
    main()
