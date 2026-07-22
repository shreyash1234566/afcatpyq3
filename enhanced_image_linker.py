"""
Enhanced Image Linker - Second Pass
====================================
Tries harder to match questions to PDF pages for questions where
the first pass couldn't find the page.

Uses fuzzy text matching on question content rather than just question numbers.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import fitz
except ImportError:
    print("[ERROR] PyMuPDF required. pip install PyMuPDF")
    exit(1)

BASE_DIR = Path(r"e:\afcatpyq3")
PAPERS_DIR = BASE_DIR / "data" / "papers"
IMAGES_DIR = BASE_DIR / "data" / "images"
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"


def clean_text(text):
    """Normalize text for fuzzy matching."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text


def get_pdf_page_texts(pdf_path):
    """Get cleaned text for each page of a PDF."""
    doc = fitz.open(str(pdf_path))
    pages = {}
    for i in range(len(doc)):
        text = doc[i].get_text()
        pages[i + 1] = clean_text(text)
    doc.close()
    return pages


def get_pdf_page_images(pdf_path):
    """Get image info per page (without re-extracting)."""
    doc = fitz.open(str(pdf_path))
    page_images = defaultdict(int)
    for i in range(len(doc)):
        imgs = doc[i].get_images(full=True)
        # Count significant images only
        for img in imgs:
            try:
                base_img = doc.extract_image(img[0])
                if base_img and base_img.get("width", 0) >= 20 and base_img.get("height", 0) >= 20:
                    page_images[i + 1] += 1
            except:
                pass
    doc.close()
    return page_images


def find_page_fuzzy(question_text, page_texts, question_number):
    """
    Find the page containing a question using fuzzy text matching.
    Tries multiple strategies.
    """
    clean_q = clean_text(question_text)

    # Strategy 1: Look for unique substrings from the question (first 50 chars)
    # Take multiple chunks of the question text
    chunks = []
    words = clean_q.split()
    if len(words) >= 5:
        chunks.append(' '.join(words[:5]))
        chunks.append(' '.join(words[:8]))
    if len(words) >= 3:
        chunks.append(' '.join(words[:3]))

    best_page = None
    best_score = 0

    for page_num, page_text in page_texts.items():
        score = 0
        for chunk in chunks:
            if chunk in page_text:
                score += len(chunk)

        if score > best_score:
            best_score = score
            best_page = page_num

    # Require at least a reasonable match
    if best_score >= 15:
        return best_page

    # Strategy 2: Search for question number followed by nearby text
    q_num = str(question_number)
    for page_num, page_text in page_texts.items():
        # Check if question number appears and some question text is nearby
        if q_num in page_text:
            if len(words) >= 3 and ' '.join(words[:3]) in page_text:
                return page_num

    return None


def main():
    print("=" * 70)
    print("  ENHANCED IMAGE LINKER - SECOND PASS")
    print("=" * 70)

    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find questions that need linking
    unlinked = [
        (i, q) for i, q in enumerate(data)
        if q.get("image_source", {}).get("status") == "page_not_found"
    ]

    print(f"\n  Questions needing page matching: {len(unlinked)}")

    if not unlinked:
        print("  No unlinked questions to process!")
        return

    # Group by PDF
    pdf_groups = defaultdict(list)
    for idx, q in unlinked:
        pdf_groups[q["file_name"]].append((idx, q))

    newly_linked = 0
    still_unlinked = 0

    for pdf_name, questions in sorted(pdf_groups.items()):
        pdf_path = PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            continue

        print(f"\n  Processing: {pdf_name} ({len(questions)} unlinked)")

        # Get page texts and image info
        page_texts = get_pdf_page_texts(pdf_path)
        page_img_counts = get_pdf_page_images(pdf_path)

        pdf_stem = Path(pdf_name).stem
        img_dir = IMAGES_DIR / pdf_stem

        # Get existing extracted image paths
        existing_images = defaultdict(list)
        if img_dir.exists():
            for img_file in img_dir.glob("*"):
                # Parse page number from filename like p5_img1.png
                match = re.match(r'p(\d+)_img\d+\.', img_file.name)
                if match:
                    page = int(match.group(1))
                    existing_images[page].append(
                        str(img_file.relative_to(BASE_DIR))
                    )

        for idx, q in questions:
            q_text = q.get("question_text", "")
            q_num = q.get("question_number")

            page = find_page_fuzzy(q_text, page_texts, q_num)

            if page:
                # Check if we have images for this page or nearby pages
                linked_images = []
                for p in [page - 1, page, page + 1]:
                    if p in existing_images:
                        linked_images.extend(existing_images[p])

                if linked_images:
                    data[idx]["image_path"] = linked_images
                    data[idx]["image_source"]["status"] = "extracted"
                    data[idx]["image_source"]["page"] = page
                    data[idx]["image_source"]["image_count"] = len(linked_images)
                    newly_linked += 1
                else:
                    data[idx]["image_source"]["status"] = "page_found_no_nearby_images"
                    data[idx]["image_source"]["page"] = page
                    still_unlinked += 1
            else:
                still_unlinked += 1

    # Save
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*60}")
    print(f"  SECOND PASS RESULTS")
    print(f"  {'='*60}")
    print(f"  Newly linked: {newly_linked}")
    print(f"  Still unlinked: {still_unlinked}")

    # Final status summary
    statuses = defaultdict(int)
    for q in data:
        src = q.get("image_source")
        if src:
            statuses[src.get("status", "?")] += 1

    print(f"\n  Final status breakdown (all figure questions):")
    for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}  {status}")

    print(f"\n  [OK] Updated Q.json")


if __name__ == "__main__":
    main()
