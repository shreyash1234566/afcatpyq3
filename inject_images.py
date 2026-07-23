"""
Inject Images into Full Question Bank (data.js)
================================================
The dashboard uses:
- data.js -> question_bank (2877 questions) for topic practice mode
- data.js -> mock_test.all_questions (100 questions) for mock test mode

This script injects image_path and image_dark into BOTH,
matching questions by (question_number + file_name) which is unique.
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(r"e:\afcatpyq3")
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"
OUTPUT_DIR = BASE_DIR / "output" / "predictions_2026"
IMAGES_SRC = BASE_DIR / "data" / "images"
IMAGES_DEST = OUTPUT_DIR / "images"


def main():
    print("=" * 70)
    print("  INJECT IMAGES v3 - FULL QUESTION BANK")
    print("=" * 70)

    # 1. Copy fresh images
    print(f"\n  Copying images to {IMAGES_DEST}...")
    if IMAGES_DEST.exists():
        shutil.rmtree(IMAGES_DEST)
    shutil.copytree(IMAGES_SRC, IMAGES_DEST)
    img_count = len(list(IMAGES_DEST.rglob("*.*")))
    print(f"    [OK] {img_count} images copied.")

    # 2. Build precise mapping from Q.json
    print("\n  Building image mappings from Q.json...")
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    import re
    def clean_text(t):
        return re.sub(r'\W+', '', (t or '').lower())[:40]

    # Map by multiple keys for robustness
    img_map_by_file = {}
    img_map_by_text = {}
    for q in q_data:
        if not q.get("has_figure"):
            continue
        image_path = q.get("image_path", [])
        if not image_path:
            continue

        image_dark = q.get("image_dark", False)

        # Convert paths to be relative to the HTML file in output/predictions_2026/
        new_paths = []
        for p in image_path:
            p_clean = p.replace("\\", "/")
            # p like "data/images/AFCAT_2011.../p16_q72_f1.png"
            # dest is "output/predictions_2026/images/AFCAT_2011.../p16_q72_f1.png"
            # URL from HTML: "./images/AFCAT_2011.../p16_q72_f1.png"
            parts = p_clean.split("/")
            if "images" in parts:
                idx = parts.index("images")
                url = "./images/" + "/".join(parts[idx + 1:])
                new_paths.append(url)

        if not new_paths:
            continue

        # Primary key: (question_number, file_name)
        qnum = q.get("question_number")
        key_file = (qnum, q.get("file_name", ""))
        key_text = (qnum, clean_text(q.get("question_text")))
        
        match_data = {
            "has_figure": True,
            "image_path": new_paths,
            "image_dark": image_dark,
        }
        img_map_by_file[key_file] = match_data
        if key_text[1]:  # only use text key if there is actual text
            img_map_by_text[key_text] = match_data

    print(f"    [OK] {len(img_map_by_file)} image mappings built.")

    # 3. Load data.js
    print("\n  Loading data.js...")
    data_js_path = OUTPUT_DIR / "data.js"
    with open(data_js_path, "r", encoding="utf-8") as f:
        content = f.read()

    json_str = content.strip()
    if json_str.startswith("const dashboardData = "):
        json_str = json_str[len("const dashboardData = "):]
    if json_str.endswith(";"):
        json_str = json_str[:-1]

    dash_data = json.loads(json_str)

    # 4. Patch question_bank (2877 questions)
    print("\n  Patching question_bank...")
    qb = dash_data.get("question_bank", [])
    qb_updated = 0
    qb_cleared = 0
    for q in qb:
        qnum = q.get("question_number")
        key_file = (qnum, q.get("file_name", ""))
        key_text = (qnum, clean_text(q.get("question_text") or q.get("question")))
        
        match = img_map_by_file.get(key_file) or img_map_by_text.get(key_text)
        if match:
            q["has_figure"] = match["has_figure"]
            q["image_path"] = match["image_path"]
            q["image_dark"] = match["image_dark"]
            qb_updated += 1
        else:
            # Clear any stale old image data
            if "image_path" in q:
                del q["image_path"]
            if "image_dark" in q:
                del q["image_dark"]
            if q.get("has_figure"):
                qb_cleared += 1

    print(f"    [OK] {qb_updated} questions updated, {qb_cleared} has_figure questions with no image yet.")

    # 5. Patch mock_test.all_questions (100 questions)
    print("\n  Patching mock_test.all_questions...")
    all_q = dash_data.get("mock_test", {}).get("all_questions", [])
    mt_updated = 0
    for q in all_q:
        qnum = q.get("question_number")
        key_file = (qnum, q.get("file_name", ""))
        key_text = (qnum, clean_text(q.get("question_text") or q.get("question")))
        
        match = img_map_by_file.get(key_file) or img_map_by_text.get(key_text)
        if match:
            q["has_figure"] = match["has_figure"]
            q["image_path"] = match["image_path"]
            q["image_dark"] = match["image_dark"]
            mt_updated += 1
        else:
            if "image_path" in q:
                del q["image_path"]
            if "image_dark" in q:
                del q["image_dark"]

    print(f"    [OK] {mt_updated} mock test questions updated.")

    # 6. Write minified data.js
    new_content = "const dashboardData = " + json.dumps(dash_data, separators=(',', ':'), ensure_ascii=False) + ";\n"
    with open(data_js_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    size_mb = len(new_content) / 1024 / 1024
    print(f"\n  data.js written: {size_mb:.2f} MB")

    print("\n  [DONE] Run: git add . ; git commit ; git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
