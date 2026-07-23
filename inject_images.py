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
    #    Key: (question_number, file_name) -> {image_path, image_dark}
    print("\n  Building image mappings from Q.json...")
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    # Map by (qnum, file_name) - most reliable unique key
    img_map = {}
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
        key = (q.get("question_number"), q.get("file_name", ""))
        img_map[key] = {
            "has_figure": True,
            "image_path": new_paths,
            "image_dark": image_dark,
        }

    print(f"    [OK] {len(img_map)} image mappings built.")

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
        key = (q.get("question_number"), q.get("file_name", ""))
        match = img_map.get(key)
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
        key = (q.get("question_number"), q.get("file_name", ""))
        match = img_map.get(key)
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

    # 7. Patch index.html — idempotent image rendering block
    print("\n  Patching index.html...")
    html_path = OUTPUT_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Remove any old image patch block
    import re
    html = re.sub(
        r'/\* IMAGE_PATCH_START \*/.*?/\* IMAGE_PATCH_END \*/',
        '',
        html,
        flags=re.DOTALL
    )

    IMG_SNIPPET = """/* IMAGE_PATCH_START */
                ${(q.has_figure && q.image_path && q.image_path.length > 0) ? `
                <div style="margin:12px 0 24px 0;padding:12px;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;display:flex;flex-wrap:wrap;gap:10px;justify-content:center;align-items:center;">
                    ${q.image_path.map((imgSrc,ii) => `<img src="${imgSrc}" alt="Figure ${ii+1}" loading="lazy" style="${q.image_dark ? 'filter:invert(1) brightness(0.85);background:#111;' : 'background:#fff;'} max-height:180px;max-width:100%;object-fit:contain;border-radius:8px;padding:4px;" onerror="this.style.display='none'"/>`).join('')}
                </div>` : ''}/* IMAGE_PATCH_END */"""

    TARGET = '<p class="text-xl font-bold text-brand-navy mb-10 leading-relaxed font-heading">${q.question_text || q.question}</p>'
    if TARGET in html:
        html = html.replace(TARGET, TARGET + "\n                " + IMG_SNIPPET)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print("    [OK] index.html patched with image rendering.")
    else:
        # already patched previously with slightly different spacing - search for q.question_text pattern
        if "q.question_text || q.question" in html:
            print("    [!] Could not find exact target - HTML may have been modified. Skipping HTML patch.")
        else:
            print("    [!] Target line not found in index.html.")

    print("\n  [DONE] Run: git add . ; git commit ; git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
