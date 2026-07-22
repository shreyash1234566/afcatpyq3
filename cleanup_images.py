"""
Clean up extracted images by removing watermarks/headers/footers.
Images that appear on every page with the same file size are likely watermarks.
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(r"e:\afcatpyq3")
IMAGES_DIR = BASE_DIR / "data" / "images"
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"

def main():
    print("=" * 70)
    print("  WATERMARK / DUPLICATE IMAGE CLEANUP")
    print("=" * 70)

    total_removed = 0
    total_kept = 0

    for pdf_dir in sorted(IMAGES_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue

        # Count file sizes across pages
        size_counts = Counter()
        size_files = defaultdict(list)

        all_files = list(pdf_dir.glob("*"))
        if not all_files:
            continue

        # Count how many unique pages we have
        pages = set()
        for f in all_files:
            import re
            m = re.match(r'p(\d+)_', f.name)
            if m:
                pages.add(int(m.group(1)))

        num_pages = len(pages)

        for f in all_files:
            if f.is_file():
                sz = f.stat().st_size
                size_counts[sz] += 1
                size_files[sz].append(f)

        # If a file size appears on 60%+ of pages, it's likely a watermark
        threshold = max(3, int(num_pages * 0.6))
        watermark_sizes = {sz for sz, count in size_counts.items() if count >= threshold}

        if watermark_sizes:
            removed = 0
            for sz in watermark_sizes:
                for f in size_files[sz]:
                    f.unlink()
                    removed += 1
            total_removed += removed
            remaining = len(all_files) - removed
            total_kept += remaining
            print(f"  {pdf_dir.name}: removed {removed} watermarks, {remaining} actual images kept")
        else:
            total_kept += len(all_files)

    print(f"\n  Total removed: {total_removed}")
    print(f"  Total kept: {total_kept}")

    # Update Q.json image_path to remove references to deleted files
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_paths = 0
    for q in data:
        if q.get("image_path"):
            original = q["image_path"]
            q["image_path"] = [
                p for p in original
                if (BASE_DIR / p).exists()
            ]
            if len(q["image_path"]) < len(original):
                cleaned_paths += 1
            if q.get("image_source"):
                q["image_source"]["image_count"] = len(q["image_path"])

    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Updated image_path for {cleaned_paths} questions")
    print(f"\n  [OK] Cleanup done!")


if __name__ == "__main__":
    main()
