import fitz
import cv2
import numpy as np
import json
from pathlib import Path
import re
import os

BASE = Path(r'e:\afcatpyq3')
Q_JSON = BASE / 'data/processed/Q.json'
PAPERS_DIR = BASE / 'data/papers'
TEST_OUT = BASE / 'scratch_imgs'
os.makedirs(TEST_OUT, exist_ok=True)

def find_question_positions(doc):
    positions = {}
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans: continue
                line_text = "".join(s["text"] for s in spans).strip()
                m = re.match(r'^Q?\.?\s*(\d+)\s*[.)]\s', line_text)
                if m:
                    qnum = int(m.group(1))
                    if qnum not in positions:
                        positions[qnum] = (page_idx, line["bbox"][1])
    return positions

def render_region(doc, page_idx, y_top, y_bottom, zoom=2.5):
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_idx]
    clip = fitz.Rect(0, y_top, page.rect.width, y_bottom)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def remove_colored_regions(img_bgr):
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([180, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask_expanded = cv2.dilate(mask_color, kernel, iterations=2)
    result[mask_expanded > 0] = [255, 255, 255]
    return result

def crop_to_content(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    h_img, w_img = img_bgr.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + 2*pad)
    h = min(h_img - y, h + 2*pad)
    return img_bgr[y:y+h, x:x+w]

doc = fitz.open(str(PAPERS_DIR / 'AFCAT_2017_Memory.pdf'))
pos = find_question_positions(doc)
for qnum in [34, 35, 38, 40, 45]:
    if qnum in pos:
        page, y = pos[qnum]
        next_page, next_y = pos.get(qnum + 1, (page, doc[page].rect.height))
        if page == next_page:
            img = render_region(doc, page, y, next_y)
            cleaned = remove_colored_regions(img)
            cropped = crop_to_content(cleaned)
            if cropped is not None:
                cv2.imwrite(str(TEST_OUT / f'test_q{qnum}.png'), cropped)
                print(f"Q{qnum} extracted successfully.")
            else:
                print(f"Q{qnum} cropped to None.")
        else:
            print(f"Q{qnum} crosses page boundary.")
