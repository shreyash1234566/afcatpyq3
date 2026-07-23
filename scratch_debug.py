"""
Root cause analysis: Why do 190 has_figure questions have no images?
"""
import json, fitz
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(r"e:\afcatpyq3")
PAPERS = BASE / "data/papers"
data = json.load(open(BASE / "data/processed/Q.json", encoding="utf-8"))

fig_q = [q for q in data if q.get("has_figure")]
with_img = [q for q in fig_q if q.get("image_path")]
without_img = [q for q in fig_q if not q.get("image_path")]

print(f"has_figure total: {len(fig_q)}")
print(f"  with images:    {len(with_img)}")
print(f"  WITHOUT images: {len(without_img)}")
print()

# Classify the 190 missing ones into buckets
buckets = defaultdict(list)

for q in without_img:
    pdf = q.get("file_name", "")
    qnum = q.get("question_number")
    src = q.get("image_source", {})
    
    pdf_path = PAPERS / pdf if pdf else None
    
    # Bucket 1: PDF doesn't exist in papers/
    if not pdf:
        buckets["no_file_name"].append(q)
        continue
    if not (PAPERS / pdf).exists():
        buckets["pdf_missing"].append(q)
        continue
    
    # Bucket 2: Question text not found on any page (text search failed)
    # We know text search was used, so if image_source has no page set, it failed
    page = src.get("page")
    if not page:
        buckets["text_not_found"].append(q)
        continue
    
    # Bucket 3: Text found but no images on that page region
    buckets["no_images_in_region"].append(q)

print("ROOT CAUSE BUCKETS:")
for bucket, qs in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print(f"  {bucket}: {len(qs)}")
    # Show sample
    for q in qs[:3]:
        print(f"    Q{q.get('question_number')} {q.get('file_name','')} | {q.get('topic')} | {q.get('question_text','')[:50]}")
    print()

# Deep dive into text_not_found: check if these PDFs are image-based (scanned)
print("="*60)
print("DEEP DIVE: text_not_found - are PDFs scanned/image-based?")
print("="*60)
pdf_text_quality = {}
for q in buckets["text_not_found"][:20]:
    pdf = q.get("file_name","")
    if pdf in pdf_text_quality:
        continue
    pdf_path = PAPERS / pdf
    if not pdf_path.exists():
        continue
    doc = fitz.open(str(pdf_path))
    # Check first 3 pages for text content
    total_chars = 0
    for pi in range(min(3, len(doc))):
        total_chars += len(doc[pi].get_text().strip())
    doc.close()
    pdf_text_quality[pdf] = total_chars
    print(f"  {pdf}: {total_chars} chars in first 3 pages -> {'TEXT' if total_chars > 500 else 'SCANNED/IMAGE'}")
