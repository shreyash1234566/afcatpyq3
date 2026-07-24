import fitz
import cv2
import numpy as np
from PIL import Image
import io

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
        if 30 < w < 400 and 30 < h < 400:
            aspect = w / float(h)
            if 0.5 < aspect < 2.0:
                # Check solidity/area
                area = cv2.contourArea(cnt)
                if area > 100:
                    circle_boxes.append((x, y, x+w, y+h))
                    
    if not circle_boxes:
        print(f"No circles found on {pdf_path.split('/')[-1]} page {page_idx+1}!")
        return
        
    # Get bounding box of ALL circles
    min_x = min(b[0] for b in circle_boxes)
    min_y = min(b[1] for b in circle_boxes)
    max_x = max(b[2] for b in circle_boxes)
    max_y = max(b[3] for b in circle_boxes)
    
    # Add padding
    pad = 20
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(img_array.shape[1], max_x + pad)
    max_y = min(img_array.shape[0], max_y + pad)
    
    cropped = img_array[min_y:max_y, min_x:max_x]
    
    cv2.imwrite(out_path, cropped)
    print(f"Saved perfectly cropped Venn diagrams to {out_path}")

# Test on 2014 P2 (page 13) and 2015 P2 (page 16)
extract_venn_by_circles(r'e:\afcatpyq3\data\papers\AFCAT_2014_Official_Paper2.pdf', 12, r'e:\afcatpyq3\test_venn_2014.png')
extract_venn_by_circles(r'e:\afcatpyq3\data\papers\AFCAT_2015_Official_Paper2.pdf', 15, r'e:\afcatpyq3\test_venn_2015.png')
