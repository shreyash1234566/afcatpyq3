"""
DEFINITIVE IMAGE-TO-QUESTION MAPPING
=====================================
Algorithm: "TextSearch + Vertical Bbox Partitioning"
(Best practice from Document Layout Analysis literature)

Key insight from PDF investigation:
1. Question text is searchable via page.search_for() - gives EXACT Y position  
2. Diagrams for each question appear BELOW that question's text
3. The diagrams END before the next question's text starts
4. Each answer diagram (a)(b)(c)(d) is stored as a SEPARATE small image

Algorithm:
1. For each figure question, use page.search_for(question_text_fragment) to find
   the EXACT bounding box of the question text on its PDF page
2. Find the next question's Y position (next question number on same page, or page bottom)
3. Extract ALL images whose center Y is between question_text_y and next_question_y
4. Filter out:
   - Adda247 logos (pink/dark Adda247 logo detected by checking for brand shape)
   - Watermarks (repeated identical images across pages)  
   - Too-small images (< 40x40 px)

This is essentially pdfminer's "LTFigure grouping" algorithm adapted for our use case.
Reference: PDFMiner layout analysis, Tesseract's PDF region detection
"""

import json
import fitz
from pathlib import Path
from PIL import Image
import io
from collections import defaultdict

BASE_DIR = Path(r"e:\afcatpyq3")
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"
PAPERS_DIR = BASE_DIR / "data" / "papers"
IMAGES_DIR = BASE_DIR / "data" / "images"

# Min dimensions to be a valid question figure
MIN_W, MIN_H = 40, 40

# Adda247 watermarks: known dark logos (dark bg pink logo) - detect by size ratio
# The logo appears at top of every page, ~y<100
WATERMARK_MAX_Y = 100.0  # anything at top 100px is header/watermark


def is_watermark_or_logo(rect, img_bytes, page_height):
    """Returns True if image is a header watermark/logo."""
    y0 = rect.y0
    # Top of page = header/logo zone
    if y0 < WATERMARK_MAX_Y:
        return True, "header_zone"
    # Check for Adda247 dark logo (very dark with pink tones)
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pixels = list(img.getdata())
        step = max(1, len(pixels) // 200)
        sample = pixels[::step][:200]
        total = len(sample)
        
        dark = sum(1 for r,g,b in sample if r<40 and g<40 and b<40)
        # Pink pixels (Adda247 logo color ~230,170,170)
        pink = sum(1 for r,g,b in sample if r>180 and g>120 and b>120 
                   and r>g and r>b and abs(g-b)<50)
        # Red pixels (Test Prime ad ~220,40,40)
        red = sum(1 for r,g,b in sample if r>140 and r>g*1.4 and r>b*1.4)
        
        dark_ratio = dark / total
        pink_ratio = pink / total
        red_ratio = red / total
        
        # Adda247 dark logo = very dark background + pink logo shape
        if dark_ratio > 0.5 and pink_ratio > 0.04:
            return True, f"adda247_logo (dark={dark_ratio:.0%} pink={pink_ratio:.0%})"
        
        # Red advertisement (Test Prime etc)
        if red_ratio > 0.18:
            return True, f"red_ad ({red_ratio:.0%} red)"
    except:
        pass
    
    return False, ""


def get_question_text_position(page, question_text, page_height):
    """
    Search for the question text on the page, return y0 (normalized).
    Try progressively shorter fragments until found.
    """
    # Try progressively shorter text fragments
    words = question_text.strip().split()
    
    # Build fragments of decreasing length (at least 5 words)
    for length in range(min(12, len(words)), 4, -1):
        fragment = " ".join(words[:length])
        instances = page.search_for(fragment)
        if instances:
            return instances[0].y0  # top Y of the found text
    
    return None  # not found


def find_all_question_positions_on_page(page, question_numbers, page_height):
    """
    Find Y positions of ALL question numbers on a page.
    Returns sorted list of (y_pos, question_number)
    """
    results = []
    text_dict = page.get_text("dict")
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join(s["text"] for s in spans).strip()
            y_top = line["bbox"][1]
            
            for qn in question_numbers:
                import re
                # Match "66.", "66)", "Q66", etc.
                patterns = [
                    rf"^{qn}[.)]\s*",
                    rf"^Q\.?\s*{qn}\b",
                    rf"^\({qn}\)",
                ]
                for pat in patterns:
                    if re.match(pat, line_text):
                        results.append((y_top, qn))
                        break
    
    results.sort()
    return results


def extract_images_for_question(pdf_path, page_idx, q_y_top, next_q_y, 
                                 page_height, doc=None):
    """
    Extract all images from a page that fall between q_y_top and next_q_y.
    Returns list of {bytes, ext, width, height, y_center}
    """
    close_doc = False
    if doc is None:
        doc = fitz.open(str(pdf_path))
        close_doc = True
    
    page = doc[page_idx]
    results = []
    
    all_images = page.get_images(full=True)
    seen_xrefs = set()
    
    for img_info in all_images:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        
        try:
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            
            # Image center Y
            y_center = (rect.y0 + rect.y1) / 2
            
            # Must be within question's vertical region
            if y_center < q_y_top - 5 or y_center > next_q_y + 5:
                continue
            
            base_image = doc.extract_image(xref)
            if not base_image:
                continue
            
            w = base_image["width"]
            h = base_image["height"]
            
            if w < MIN_W or h < MIN_H:
                continue
            
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            
            # Filter watermarks/logos
            is_wm, reason = is_watermark_or_logo(rect, img_bytes, page_height)
            if is_wm:
                continue
            
            results.append({
                "bytes": img_bytes,
                "ext": ext,
                "width": w,
                "height": h,
                "y_center": y_center,
                "rect": rect,
            })
        except:
            continue
    
    if close_doc:
        doc.close()
    
    # Sort top to bottom
    results.sort(key=lambda x: x["y_center"])
    return results


def main():
    print("=" * 70)
    print("  DEFINITIVE IMAGE-TO-QUESTION MAPPER")
    print("  Algorithm: TextSearch + Vertical BBox Partitioning")
    print("=" * 70)
    
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Step 1: Determine has_figure for ALL questions
    # A question needs a figure if its choices are "Diagram (a/b/c/d)" type
    # OR if it has keywords like "figure below", "following figure", "figure shown"
    FIGURE_INDICATORS = [
        "diagram (a)", "diagram (b)", "figure (a)", "figure (b)",
        "following figure", "figure below", "given figure", "answer figure",
        "figure shown", "figure given", "question figure",
    ]
    # For NV questions: if topic is a visual reasoning type and choices are lettered diagrams
    NV_TOPICS = {
        "Venn Diagrams", "Non-Verbal Series", "Non-Verbal Analogy",
        "Non-Verbal Classification", "Non-Verbal Pattern", "Non-Verbal Puzzle",
        "Non-Verbal Orientation", "Non-Verbal Spatial", "Pattern Completion",
    }
    
    newly_flagged = 0
    for q in data:
        if q.get("has_figure"):
            continue
        q_text_lower = q.get("question_text", "").lower()
        choices = q.get("choices", [])
        choices_text = " ".join(
            (c.get("text","") if isinstance(c,dict) else str(c)).lower()
            for c in choices
        )
        combined = q_text_lower + " " + choices_text
        topic = q.get("topic", "")
        
        # Flag if any figure indicator present in text/choices
        if any(ind in combined for ind in FIGURE_INDICATORS):
            q["has_figure"] = True
            if not q.get("image_source"):
                q["image_source"] = {}
            newly_flagged += 1
        # Flag if it's a visual reasoning topic (Venn etc) - always has diagrams
        elif topic in NV_TOPICS:
            q["has_figure"] = True
            if not q.get("image_source"):
                q["image_source"] = {}
            newly_flagged += 1
    
    print(f"\n  Step 1: Newly flagged as has_figure: {newly_flagged}")
    all_fig_q = [q for q in data if q.get("has_figure")]
    print(f"  Total figure questions: {len(all_fig_q)}")
    
    # Step 2: Clear old image_path for all figure questions  
    for q in data:
        if q.get("has_figure"):
            q["image_path"] = []
            q["image_dark"] = False
    
    # Step 3: Group by PDF file (process one PDF at a time for efficiency)
    pdf_questions = defaultdict(list)
    for i, q in enumerate(data):
        if not q.get("has_figure"):
            continue
        pdf_name = q.get("file_name", "")
        if pdf_name:
            pdf_questions[pdf_name].append((i, q))
    
    print(f"\n  Step 2: Processing {len(pdf_questions)} PDFs...")
    
    precisely_matched = 0
    text_search_matched = 0
    fallback_matched = 0
    no_images = 0
    
    for pdf_name, q_list in pdf_questions.items():
        pdf_path = PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            continue
        
        doc = fitz.open(str(pdf_path))
        pdf_stem = pdf_name.replace(".pdf", "")
        img_dir = IMAGES_DIR / pdf_stem
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Group questions by page
        # First pass: try to find page via text search
        for idx, q in q_list:
            q_text = q.get("question_text", "")
            qnum = q.get("question_number", 0)
            
            # Find page via text search
            found_page = None
            found_y = None
            
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                y = get_question_text_position(page, q_text, page.rect.height)
                if y is not None:
                    found_page = page_idx
                    found_y = y
                    break
            
            # If text search fails, fallback to searching for the question number
            if found_page is None:
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    q_positions = find_all_question_positions_on_page(page, [qnum], page.rect.height)
                    if q_positions:
                        found_page = page_idx
                        found_y = q_positions[0][0]
                        break
                        
            if found_page is not None:
                # Store in image_source for next time
                if not q.get("image_source"):
                    q["image_source"] = {}
                q["image_source"]["page"] = found_page + 1
                q["image_source"]["pdf"] = pdf_name
                q["image_source"]["q_y"] = found_y
            
            if found_page is None:
                # Try with stored page from previous extraction as absolute last resort
                src = q.get("image_source", {})
                stored_page = src.get("page")
                if stored_page:
                    found_page = stored_page - 1
                    found_y = 0  # fallback - scan whole page
            
            if found_page is None:
                no_images += 1
                continue
            
            page = doc[found_page]
            page_h = page.rect.height
            
            # Find next question boundary on same page
            # Look for any other figure questions on this same page
            same_page_qs = [(other_idx, other_q) for other_idx, other_q in q_list
                           if (other_q.get("image_source", {}).get("page") == found_page + 1
                               and other_q.get("question_number", 0) > qnum)]
            
            # Also find question numbers of ALL questions on this page
            all_q_on_page = [i for i in range(max(1, qnum-2), qnum+10)]
            q_positions = find_all_question_positions_on_page(page, all_q_on_page, page_h)
            
            # Find next question's Y (the question that comes after current one)
            next_q_y = page_h  # default = bottom of page
            current_q_pos = None
            
            for y_pos, q_num in q_positions:
                if q_num == qnum:
                    current_q_pos = y_pos
                elif current_q_pos is not None and q_num > qnum:
                    next_q_y = y_pos
                    break
            
            # If we found the question text, use it as the search start
            q_search_y = found_y if found_y is not None else (current_q_pos or 0)
            
            # Extract images in this question's region
            imgs = extract_images_for_question(
                pdf_path, found_page,
                q_search_y, next_q_y,
                page_h, doc=doc
            )
            
            if not imgs:
                no_images += 1
                continue
            
            # Save images
            saved_paths = []
            has_dark = False
            
            for fi, img_data in enumerate(imgs):
                fname = f"p{found_page+1}_q{qnum}_f{fi+1}.{img_data['ext']}"
                fpath = img_dir / fname
                with open(fpath, "wb") as f:
                    f.write(img_data["bytes"])
                rel = str(fpath.relative_to(BASE_DIR)).replace("\\", "/")
                saved_paths.append(rel)
                
                # Check if dark
                try:
                    pil = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                    px = list(pil.getdata())
                    step = max(1, len(px)//50)
                    smp = px[::step][:50]
                    dark_r = sum(1 for r,g,b in smp if r<40 and g<40 and b<40) / len(smp)
                    if dark_r > 0.55:
                        has_dark = True
                except:
                    pass
            
            data[idx]["image_path"] = saved_paths
            data[idx]["image_dark"] = has_dark
            data[idx]["image_source"]["status"] = "extracted_textsearch"
            data[idx]["image_source"]["image_count"] = len(saved_paths)
            
            if found_y is not None:
                text_search_matched += 1
            else:
                fallback_matched += 1
        
        doc.close()
        print(f"  [{pdf_stem[:40]}] done")
    
    # Step 4: Save Q.json
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 70)
    print("  RESULTS")
    print(f"  Text-search matched: {text_search_matched}")
    print(f"  Fallback matched:    {fallback_matched}")
    print(f"  No images found:     {no_images}")
    total_with_img = len([q for q in data if q.get("image_path")])
    print(f"  Total with images:   {total_with_img}")
    print()
    print("  [DONE] Run inject_images.py next then git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
