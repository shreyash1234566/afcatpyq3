"""
Fix all PDF filename mismatches in Q.json, then re-run image extraction
for the newly mapped files.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

BASE_DIR = Path(r"e:\afcatpyq3")
PAPERS_DIR = BASE_DIR / "data" / "papers"
IMAGES_DIR = BASE_DIR / "data" / "images"
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"

# Complete mapping of mismatched filenames
PDF_NAME_MAP = {
    # Old official paper naming
    "AFCAT_2_2012.pdf": "AFCAT_2012_Official_Paper2.pdf",
    "AFCAT_1_2014.pdf": "AFCAT_2014_Official_Paper1.pdf",
    "AFCAT_2_2014.pdf": "AFCAT_2014_Official_Paper2.pdf",

    # 2021
    "Indian_Air_Force_AFCAT_2021_Memory_Based.pdf": "AFCAT_2021_Memory.pdf",

    # 2022 Feb
    "AFCAT Memory Based Paper - 13 Feb 2022": "AFCAT_2022_Feb13_Memory.pdf",
    "AFCAT Memory Based Paper - 14 Feb 2022": "AFCAT_2022_Feb14_Memory.pdf",
    "Defence_AFCAT_Memory_Based_Paper-13_Feb_2022_DPP.pdf": "DPP_2022_Feb13.pdf",
    "Defence_AFCAT_Memory_Based_Paper-_13_Feb_2022_DPP.pdf": "DPP_2022_Feb13.pdf",
    "Defence_AFCAT_Memory_Based_Paper_-13_Feb_2022_DPP.pdf": "DPP_2022_Feb13.pdf",

    # 2022 Aug
    "AFCAT_Memory_Based_Paper_26_Aug_2022_Shift_1.pdf": "AFCAT_2022_Aug26_Memory.pdf",

    # 2024 Feb
    "AFCAT Memory Based Paper - 17 Feb 2024": "AFCAT_2024_Feb17_Memory.pdf",
    "AFCAT_Memory_Based_Paper_17_Feb_2024.pdf": "AFCAT_2024_Feb17_Memory.pdf",

    # 2024 Aug
    "AFCAT MBT 9 Aug 2024": "AFCAT_2024_Aug09_Shift1_Memory.pdf",
    "AFCAT_9_Aug_2024_Shift_1.pdf": "AFCAT_2024_Aug09_Shift1_Memory.pdf",
    "AFCAT_MBT_10_Aug_shift-1_2024.pdf": "AFCAT_2024_Aug10_Shift1_Memory.pdf",
    "AFCAT_2024-2_MBT_English_Shift-1_11_Aug_2024.pdf": "AFCAT_2024_Aug11_Memory.pdf",
    "AFCAT_2024-2_MBT_English_Shift-_1_11_Aug_2024.pdf": "AFCAT_2024_Aug11_Memory.pdf",
    "AFCAT_2024_-2_MBT_English_Shift-1_11_Aug_2024.pdf": "AFCAT_2024_Aug11_Memory.pdf",

    # 2025
    "AFCAT-01_2025-Memory-Based-Paper-Held-On_-22-Jan-2025-Shift-1.pdf": "AFCAT_2025_Jan22_Shift1_Memory.pdf",
    "AFCAT_02_2025_Shift_2.pdf": "AFCAT_2025_Aug23_Shift2_Memory.pdf",
    "AFCAT_02_2025_24_Aug_Shift_1.pdf": "AFCAT_2025_Aug24_Shift1_Memory.pdf",
    "AFCAT-02_2025_-24-August-2025-Shift-2-Memory-Based-Paper.pdf": "AFCAT_2025_Aug24_Shift2_Memory.pdf",
}

actual_pdfs = {p.name for p in PAPERS_DIR.glob("*.pdf")}


def clean_text(text):
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text


MIN_IMAGE_DIM = 20
MIN_IMAGE_AREA = 1000


def extract_images_from_pdf(pdf_path, output_dir):
    """Extract images from PDF, return page->image paths dict."""
    if not HAS_FITZ:
        return {}, 0

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
                w = base_image.get("width", 0)
                h = base_image.get("height", 0)
                if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM or w * h < MIN_IMAGE_AREA:
                    continue

                img_counter += 1
                img_filename = f"p{page_num + 1}_img{img_idx + 1}.{base_image['ext']}"
                img_path = output_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])

                page_images[page_num + 1].append(str(img_path.relative_to(BASE_DIR)))
            except Exception:
                pass

    doc.close()
    return page_images, img_counter


def find_page_fuzzy(question_text, page_texts, question_number):
    """Find page using fuzzy text matching."""
    clean_q = clean_text(question_text)
    words = clean_q.split()

    best_page = None
    best_score = 0

    chunks = []
    if len(words) >= 5:
        chunks.append(' '.join(words[:5]))
        chunks.append(' '.join(words[:8]))
    if len(words) >= 3:
        chunks.append(' '.join(words[:3]))

    for page_num, page_text in page_texts.items():
        score = sum(len(c) for c in chunks if c in page_text)
        if score > best_score:
            best_score = score
            best_page = page_num

    if best_score >= 15:
        return best_page

    q_num = str(question_number)
    for page_num, page_text in page_texts.items():
        if q_num in page_text and len(words) >= 3 and ' '.join(words[:3]) in page_text:
            return page_num

    return None


def main():
    print("=" * 70)
    print("  FIX PDF FILENAMES + RE-EXTRACT IMAGES")
    print("=" * 70)

    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 1: Fix filenames
    filename_fixes = 0
    newly_mapped_pdfs = set()
    for q in data:
        fn = q.get("file_name", "")
        if fn in PDF_NAME_MAP:
            new_fn = PDF_NAME_MAP[fn]
            q["file_name"] = new_fn
            filename_fixes += 1
            if new_fn in actual_pdfs:
                newly_mapped_pdfs.add(new_fn)

    print(f"\n  Fixed {filename_fixes} filename references")
    print(f"  Newly mapped PDFs: {len(newly_mapped_pdfs)}")

    # Step 2: Extract images from newly mapped PDFs (if not already done)
    if HAS_FITZ:
        for pdf_name in sorted(newly_mapped_pdfs):
            pdf_stem = Path(pdf_name).stem
            img_dir = IMAGES_DIR / pdf_stem

            if img_dir.exists() and list(img_dir.glob("*")):
                print(f"  [SKIP] {pdf_name} - images already extracted")
                continue

            pdf_path = PAPERS_DIR / pdf_name
            print(f"  [EXTRACT] {pdf_name}...")
            page_images, count = extract_images_from_pdf(pdf_path, img_dir)
            print(f"    -> {count} images from {len(page_images)} pages")

    # Step 3: Link figure questions from newly mapped PDFs
    print("\n  Linking figure questions to images...")
    newly_linked = 0

    # Group unlinked figure questions by PDF
    unlinked_by_pdf = defaultdict(list)
    for i, q in enumerate(data):
        if q.get("has_figure") and not q.get("image_path"):
            fn = q.get("file_name", "")
            if fn in actual_pdfs:
                unlinked_by_pdf[fn].append(i)

    for pdf_name, q_indices in sorted(unlinked_by_pdf.items()):
        pdf_path = PAPERS_DIR / pdf_name
        pdf_stem = Path(pdf_name).stem
        img_dir = IMAGES_DIR / pdf_stem

        if not img_dir.exists():
            continue

        # Get existing images per page
        existing_images = defaultdict(list)
        for img_file in img_dir.glob("*"):
            match = re.match(r'p(\d+)_img\d+\.', img_file.name)
            if match:
                page = int(match.group(1))
                existing_images[page].append(str(img_file.relative_to(BASE_DIR)))

        if not existing_images:
            continue

        # Get page texts for fuzzy matching
        if HAS_FITZ:
            doc = fitz.open(str(pdf_path))
            page_texts = {}
            for i in range(len(doc)):
                page_texts[i + 1] = clean_text(doc[i].get_text())
            doc.close()
        else:
            page_texts = {}

        print(f"  Processing: {pdf_name} ({len(q_indices)} unlinked)")

        for idx in q_indices:
            q = data[idx]
            q_text = q.get("question_text", "")
            q_num = q.get("question_number")

            page = find_page_fuzzy(q_text, page_texts, q_num) if page_texts else None

            if page:
                linked_images = []
                for p in [page - 1, page, page + 1]:
                    if p in existing_images:
                        linked_images.extend(existing_images[p])

                if linked_images:
                    q["image_path"] = linked_images
                    q["image_source"] = {
                        "pdf": pdf_name,
                        "question_number": q_num,
                        "page": page,
                        "status": "extracted",
                        "image_count": len(linked_images)
                    }
                    newly_linked += 1
                else:
                    q["image_source"] = {
                        "pdf": pdf_name,
                        "question_number": q_num,
                        "page": page,
                        "status": "page_found_no_nearby_images"
                    }
            else:
                q["image_source"] = {
                    "pdf": pdf_name,
                    "question_number": q_num,
                    "status": "page_not_found"
                }

    # Save
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  {'='*60}")
    print(f"  Filename fixes applied: {filename_fixes}")
    print(f"  Newly linked in this pass: {newly_linked}")

    # Final counts
    total_fig = sum(1 for q in data if q.get("has_figure"))
    total_extracted = sum(1 for q in data if q.get("image_source", {}).get("status") == "extracted")
    total_with_path = sum(1 for q in data if q.get("image_path"))

    print(f"\n  Total figure questions: {total_fig}")
    print(f"  Questions with extracted images: {total_extracted}")
    print(f"  Questions with image_path: {total_with_path}")

    statuses = Counter(
        q.get("image_source", {}).get("status", "no_source")
        for q in data if q.get("has_figure")
    )
    print(f"\n  Status breakdown:")
    for status, count in statuses.most_common():
        print(f"    {count:4d}  {status}")

    print(f"\n  [OK] Done!")


if __name__ == "__main__":
    main()
