"""
Delete Advertisement Images from Extracted Images
===================================================
Targets:
- Adda247 "Test Prime" ads (red background)
- Any image where red is the dominant color (r > 150, r > g*1.5, r > b*1.5)
- Adda247 logos / watermarks
- After deleting, remove their paths from Q.json image_path arrays
"""

import json
import os
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path(r"e:\afcatpyq3")
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"
IMAGES_DIR = BASE_DIR / "data" / "images"

def is_advertisement(img_path):
    """
    Returns True if image looks like an ad:
    - Red dominant color (Adda247, TestBook ads)
    - Orange dominant (some coaching ads)  
    - Blue/purple promotional (some coaching institutes)
    - Text-heavy colorful banners
    """
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Sample pixels evenly
        pixels = list(img.getdata())
        step = max(1, len(pixels) // 300)
        sample = pixels[::step][:300]
        total = len(sample)
        
        # Count red-dominant pixels (Adda247 red = ~220,40,40)
        red_dominant = sum(1 for r, g, b in sample 
                          if r > 140 and r > g * 1.4 and r > b * 1.4)
        
        # Count orange-dominant pixels
        orange_dominant = sum(1 for r, g, b in sample 
                             if r > 180 and g > 80 and g < 160 and b < 80)
        
        # Count bright non-white colorful pixels (colorful banner backgrounds)
        colorful = sum(1 for r, g, b in sample 
                      if max(r,g,b) > 150 and max(r,g,b) - min(r,g,b) > 80)
        
        red_ratio = red_dominant / total
        orange_ratio = orange_dominant / total
        colorful_ratio = colorful / total
        
        # Ad if >20% red dominant pixels
        if red_ratio > 0.20:
            return True, f"red_bg ({red_ratio:.0%} red)"
        
        # Ad if >25% orange dominant
        if orange_ratio > 0.25:
            return True, f"orange_bg ({orange_ratio:.0%} orange)"
        
        # Very colorful non-question image (promotional graphic)
        # Real question figures are mostly black lines on white background
        dark_pixels = sum(1 for r, g, b in sample if r < 50 and g < 50 and b < 50)
        white_pixels = sum(1 for r, g, b in sample if r > 220 and g > 220 and b > 220)
        bw_ratio = (dark_pixels + white_pixels) / total
        
        # If image is mostly colorful (not black+white) AND not a dark scan
        if bw_ratio < 0.55 and colorful_ratio > 0.40:
            return True, f"colorful_ad (bw={bw_ratio:.0%}, colorful={colorful_ratio:.0%})"
        
        return False, "valid"
    except Exception as e:
        return False, f"error: {e}"


def main():
    print("=" * 70)
    print("  ADVERTISEMENT IMAGE CLEANER")
    print("=" * 70)
    
    # Scan all images
    all_images = list(IMAGES_DIR.rglob("*.jpeg")) + \
                 list(IMAGES_DIR.rglob("*.jpg")) + \
                 list(IMAGES_DIR.rglob("*.png"))
    
    print(f"\n  Scanning {len(all_images)} images...")
    
    ads_found = []
    for img_path in all_images:
        is_ad, reason = is_advertisement(img_path)
        if is_ad:
            ads_found.append((img_path, reason))
    
    print(f"\n  Advertisement images found: {len(ads_found)}")
    print("\n  Samples:")
    for p, r in ads_found[:10]:
        print(f"    [{r}] {p.name} ({p.parent.name})")
    
    if not ads_found:
        print("  No ads found!")
        return
    
    print(f"\n  Deleting {len(ads_found)} ad images...")
    deleted_paths = set()
    for img_path, reason in ads_found:
        rel_path = str(img_path.relative_to(BASE_DIR)).replace("\\", "/")
        deleted_paths.add(rel_path)
        deleted_paths.add(str(img_path.relative_to(BASE_DIR)))  # both slashes
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"    [!] Could not delete {img_path.name}: {e}")
    
    print(f"  [OK] Deleted {len(ads_found)} images.")
    
    # Remove deleted image paths from Q.json
    print("\n  Cleaning Q.json image_path arrays...")
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned_q = 0
    emptied_q = 0
    for q in data:
        if not q.get("image_path"):
            continue
        old_paths = q["image_path"]
        new_paths = []
        for p in old_paths:
            p_norm = p.replace("\\", "/")
            # Check if this path was deleted
            if not any(p_norm.endswith(str(dp).replace("\\", "/").split("data/images/")[-1]) 
                      for dp, _ in ads_found):
                new_paths.append(p)
        
        if len(new_paths) != len(old_paths):
            q["image_path"] = new_paths
            cleaned_q += 1
            if not new_paths:
                # No images left for this question
                q.pop("image_dark", None)
                emptied_q += 1
    
    with open(Q_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] Q.json: {cleaned_q} questions cleaned, {emptied_q} fully emptied.")
    print("\n  [DONE] Run inject_images.py + git push next.")
    print("=" * 70)


if __name__ == "__main__":
    main()
