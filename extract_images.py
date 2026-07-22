"""
AFCAT PDF Image Extractor
=========================
Extracts images from AFCAT PDF question papers and links them to questions in Q.json.

For each question with has_figure=true, this script:
1. Opens the source PDF (identified by file_name field)
2. Extracts images from the relevant pages
3. Saves them to data/images/<pdf_name>/
4. Updates Q.json with image_path field

Usage: python extract_images.py
Requirements: pip install PyMuPDF (fitz)
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("[WARN] PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
    print("       Running in DRY-RUN mode (will mark questions but not extract images).")


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(r"e:\afcatpyq3")
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
IMAGES_DIR = DATA_DIR / "images"
Q_JSON_PATH = DATA_DIR / "processed" / "Q.json"

# Minimum image size (width * height) to avoid extracting tiny icons/artifacts
MIN_IMAGE_AREA = 1000  # pixels
MIN_IMAGE_DIM = 20     # minimum width or height in pixels


def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extract all images from a PDF file.
    Returns dict: page_number -> list of image paths
    """
    if not HAS_FITZ:
        return {}

    doc = fitz.open(str(pdf_path))
    page_images = defaultdict(list)

    output_dir.mkdir(parents=True, exist_ok=True)

    img_counter = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip tiny images (logos, bullets, etc.)
                if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
                    continue
                if width * height < MIN_IMAGE_AREA:
                    continue

                img_counter += 1
                img_filename = f"p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                img_path = output_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                page_images[page_num + 1].append({
                    "path": str(img_path.relative_to(BASE_DIR)),
                    "width": width,
                    "height": height,
                    "page": page_num + 1,
                })
            except Exception as e:
                print(f"    [!] Error extracting image xref={xref} from page {page_num+1}: {e}")

    doc.close()
    return page_images, img_counter


def extract_page_text(pdf_path, page_num):
    """Extract text from a specific PDF page for matching questions."""
    if not HAS_FITZ:
        return ""
    doc = fitz.open(str(pdf_path))
    if page_num - 1 < len(doc):
        text = doc[page_num - 1].get_text()
        doc.close()
        return text
    doc.close()
    return ""


def find_question_page(pdf_path, question_number, question_text):
    """
    Find which page a question appears on by searching for the question number
    and/or question text in the PDF pages.
    """
    if not HAS_FITZ:
        return None

    doc = fitz.open(str(pdf_path))
    q_num_str = str(question_number)

    # Try to find the page containing this question number
    best_page = None
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()

        # Look for question number patterns like "Q.1" "1." "Q1" etc.
        patterns = [
            rf'\b{q_num_str}\.',
            rf'Q\.?\s*{q_num_str}\b',
            rf'Question\s*{q_num_str}\b',
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                # Verify with some question text if available
                if question_text:
                    # Check first 30 chars of question text
                    clean_q = question_text[:30].strip()
                    if clean_q and clean_q[:15] in text:
                        best_page = page_num + 1
                        break
                else:
                    best_page = page_num + 1
                    break

        if best_page:
            break

    doc.close()
    return best_page


def main():
    print("=" * 70)
    print("  AFCAT PDF IMAGE EXTRACTOR")
    print("=" * 70)

    # Load Q.json
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group questions by PDF file
    pdf_questions = defaultdict(list)
    figure_count = 0
    for i, q in enumerate(data):
        if q.get("has_figure"):
            figure_count += 1
            pdf_questions[q.get("file_name", "")].append(i)

    print(f"\n  Total questions with has_figure=true: {figure_count}")
    print(f"  Spread across {len(pdf_questions)} PDF files")

    # Check which PDFs exist
    available_pdfs = set(p.name for p in PAPERS_DIR.glob("*.pdf"))
    matched = 0
    unmatched_pdfs = []
    for pdf_name in pdf_questions:
        if pdf_name in available_pdfs:
            matched += 1
        else:
            unmatched_pdfs.append(pdf_name)

    print(f"  PDFs found in papers/: {matched}/{len(pdf_questions)}")
    if unmatched_pdfs:
        print(f"  [!] Missing PDFs: {unmatched_pdfs[:5]}{'...' if len(unmatched_pdfs)>5 else ''}")

    if not HAS_FITZ:
        print("\n  [DRY RUN] PyMuPDF not available. Skipping image extraction.")
        print("  [DRY RUN] Adding image_source metadata to Q.json instead.\n")

        # Still add metadata about which PDF + question number to look at
        updates = 0
        for pdf_name, q_indices in pdf_questions.items():
            for idx in q_indices:
                q = data[idx]
                if pdf_name in available_pdfs:
                    q["image_source"] = {
                        "pdf": pdf_name,
                        "question_number": q.get("question_number"),
                        "status": "pending_extraction"
                    }
                else:
                    q["image_source"] = {
                        "pdf": pdf_name,
                        "question_number": q.get("question_number"),
                        "status": "pdf_not_found"
                    }
                updates += 1

        # Save
        with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Added image_source metadata to {updates} questions")
        return

    # === FULL EXTRACTION MODE ===
    print("\n  Starting image extraction...\n")

    total_images = 0
    total_linked = 0
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_name, q_indices in sorted(pdf_questions.items()):
        if pdf_name not in available_pdfs:
            continue

        pdf_path = PAPERS_DIR / pdf_name
        pdf_stem = Path(pdf_name).stem
        output_dir = IMAGES_DIR / pdf_stem

        print(f"  Processing: {pdf_name} ({len(q_indices)} figure questions)")

        # Extract all images from this PDF
        page_images, img_count = extract_images_from_pdf(pdf_path, output_dir)
        total_images += img_count

        if not page_images:
            print(f"    -> No significant images found")
            for idx in q_indices:
                data[idx]["image_source"] = {
                    "pdf": pdf_name,
                    "question_number": data[idx].get("question_number"),
                    "status": "no_images_in_pdf"
                }
            continue

        print(f"    -> Extracted {img_count} images from {len(page_images)} pages")

        # Try to link questions to their page images
        for idx in q_indices:
            q = data[idx]
            q_num = q.get("question_number")
            q_text = q.get("question_text", "")

            # Try to find which page this question is on
            page = find_question_page(pdf_path, q_num, q_text)

            if page and page in page_images:
                # Link the images from this page
                q["image_path"] = [img["path"] for img in page_images[page]]
                q["image_source"] = {
                    "pdf": pdf_name,
                    "question_number": q_num,
                    "page": page,
                    "status": "extracted",
                    "image_count": len(page_images[page])
                }
                total_linked += 1
            elif page:
                # Found the page but no images on that specific page
                # Try adjacent pages (question might span pages, or image on next page)
                nearby_images = []
                for p in [page - 1, page, page + 1]:
                    if p in page_images:
                        nearby_images.extend(page_images[p])

                if nearby_images:
                    q["image_path"] = [img["path"] for img in nearby_images]
                    q["image_source"] = {
                        "pdf": pdf_name,
                        "question_number": q_num,
                        "page": page,
                        "status": "extracted_nearby",
                        "image_count": len(nearby_images)
                    }
                    total_linked += 1
                else:
                    q["image_source"] = {
                        "pdf": pdf_name,
                        "question_number": q_num,
                        "page": page,
                        "status": "page_found_no_images"
                    }
            else:
                # Couldn't find the page - link all images from the PDF as candidates
                all_images = []
                for p_imgs in page_images.values():
                    all_images.extend(p_imgs)
                q["image_source"] = {
                    "pdf": pdf_name,
                    "question_number": q_num,
                    "status": "page_not_found",
                    "total_pdf_images": len(all_images)
                }

    # Save updated Q.json
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*60}")
    print(f"  EXTRACTION SUMMARY")
    print(f"  {'='*60}")
    print(f"  Total images extracted: {total_images}")
    print(f"  Questions linked to images: {total_linked}/{figure_count}")

    # Count statuses
    statuses = defaultdict(int)
    for q in data:
        if q.get("image_source"):
            statuses[q["image_source"].get("status", "unknown")] += 1

    print(f"\n  Status breakdown:")
    for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}  {status}")

    print(f"\n  Images saved to: {IMAGES_DIR}")
    print(f"  [OK] Updated Q.json with image_path and image_source fields")


if __name__ == "__main__":
    main()
