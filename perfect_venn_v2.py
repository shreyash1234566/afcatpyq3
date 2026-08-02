"""
Per-Question Venn Diagram Extractor v2
======================================
Each Venn Diagram question in AFCAT has its OWN unique set of 4 option diagrams (a)(b)(c)(d).
This script:
  1. Finds each question's exact region in the PDF (from Q text to next Q text)
  2. Renders that region as a high-res pixmap
  3. Uses HoughCircles to detect the Venn circles
  4. Crops tightly around ONLY the circles (no text, no watermarks)
  5. Saves a per-question image
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
                        positions[qnum] = (page_idx, line["bbox"][1])
    return positions


def find_option_labels(doc, page_idx, y_start, y_end):
    """Find Y positions of (a), (b), (c), (d) on a page within a Y range."""
    page = doc[page_idx]
    text_dict = page.get_text("dict")
    labels = {}
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            y = line["bbox"][1]
            if y < y_start or y > y_end:
                continue
            for span in line.get("spans", []):
                t = span["text"].strip().lower()
                if t in ["(a)", "a)", "(a"]:
                    labels.setdefault("a", y)
                elif t in ["(b)", "b)", "(b"]:
                    labels.setdefault("b", y)
                elif t in ["(c)", "c)", "(c", "q(c)"]:
                    labels.setdefault("c", y)
                elif t in ["(d)", "d)", "(d"]:
                    labels.setdefault("d", y)
    return labels


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
    Venn diagrams are BLACK lines on WHITE background (zero saturation).
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


def crop_circles_only(img_bgr):
    """
    Detect Venn diagram circles and crop tightly around them.
    Steps:
      1. Remove colored watermarks/ads (keep only black-on-white content)
      2. Detect circles via HoughCircles
      3. Crop tightly around detected circles
    """
    h, w = img_bgr.shape[:2]
    if h < 30 or w < 30:
        return None
    
    # Step 1: Clean the image (remove colored ads/watermarks)
    cleaned = remove_colored_regions(img_bgr)
    
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    # Step 2: Try HoughCircles with various sensitivities
    best_circles = None
    best_count = 0
    
    for param2 in [35, 25, 18, 12]:
        for min_r, max_r in [(20, 120), (15, 150), (10, 180)]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=20,
                param1=80, param2=param2,
                minRadius=min_r, maxRadius=max_r
            )
            if circles is not None:
                count = len(circles[0])
                if count > best_count and count <= 50:
                    best_circles = circles[0]
                    best_count = count
                if count >= 4:
                    break
        if best_count >= 4:
            break
    
    if best_circles is not None and best_count >= 2:
        cs = np.int32(np.around(best_circles))
        
        min_x = int(min(int(c[0]) - int(c[2]) for c in cs))
        min_y = int(min(int(c[1]) - int(c[2]) for c in cs))
        max_x = int(max(int(c[0]) + int(c[2]) for c in cs))
        max_y = int(max(int(c[1]) + int(c[2]) for c in cs))
        
        pad = 20
        min_x = max(0, min_x - pad)
        min_y = max(0, min_y - pad)
        max_x = min(w, max_x + pad)
        max_y = min(h, max_y + pad)
        
        # Crop from the CLEANED image (no watermarks)
        cropped = cleaned[min_y:max_y, min_x:max_x]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30:
            return cropped
    
    # Fallback: contour detection with circularity filter on cleaned image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_rects = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 20 or ch < 20 or cw > 350 or ch > 350:
            continue
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.35:
            circle_rects.append((x, y, x + cw, y + ch))
    
    if len(circle_rects) >= 2:
        min_x = min(r[0] for r in circle_rects) - 15
        min_y = min(r[1] for r in circle_rects) - 15
        max_x = max(r[2] for r in circle_rects) + 15
        max_y = max(r[3] for r in circle_rects) + 15
        
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w, max_x)
        max_y = min(h, max_y)
        
        cropped = cleaned[min_y:max_y, min_x:max_x]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30:
            return cropped
    
    return None


def main():
    q_data = json.load(open(Q_JSON_PATH, encoding='utf-8'))
    
    venn_by_paper = defaultdict(list)
    for q in q_data:
        if q.get('topic') == 'Venn Diagrams':
            venn_by_paper[q.get('file_name')].append(q)
    
    total_ok = 0
    total_fail = 0
    
    for paper, qs in venn_by_paper.items():
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
        print(f"  {paper} ({len(qs)} Venn Qs)")
        print(f"{'='*60}")
        
        for q in qs:
            qnum = int(q.get('question_number'))
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
                # Try +2
                if (qnum + 2) in all_pos:
                    next_page_idx, next_y = all_pos[qnum + 2]
            
            # --- Render the question's region ---
            page = doc[page_idx]
            
            if next_page_idx is None:
                # No next question found → crop to page bottom
                img_bgr = render_region(doc, page_idx, y_start, page.rect.height)
            elif next_page_idx == page_idx:
                # Same page → crop between questions
                img_bgr = render_region(doc, page_idx, y_start, next_y)
            else:
                # Cross page → stitch
                img_bgr = render_cross_page(doc, page_idx, y_start, next_page_idx, next_y)
            
            if img_bgr is None or img_bgr.shape[0] < 30:
                print(f"  Q{qnum}: [FAIL] Render too small")
                q['image_path'] = []
                q['has_figure'] = False
                total_fail += 1
                continue
            
            # --- Crop ONLY the circles ---
            cropped = crop_circles_only(img_bgr)
            
            if cropped is not None:
                out_fname = f"venn_q{qnum}.png"
                out_path = img_dir / out_fname
                cv2.imwrite(str(out_path), cropped)
                
                rel_path = f"data/images/{pdf_stem}/{out_fname}"
                q['image_path'] = [rel_path]
                q['has_figure'] = True
                q['image_dark'] = False
                total_ok += 1
                print(f"  Q{qnum}: [OK] {cropped.shape[1]}x{cropped.shape[0]} px")
            else:
                print(f"  Q{qnum}: [FAIL] No circles detected")
                q['image_path'] = []
                q['has_figure'] = False
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
