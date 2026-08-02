import json
import os
import glob
from pathlib import Path

ROOT = Path("e:/afcatpyq3")
Q_CLEAN_PATH = ROOT / "data" / "processed" / "Q_clean.json"
IMG_ROOT = ROOT / "data" / "images"

def main():
    if not Q_CLEAN_PATH.exists():
        print("Q_clean.json not found!")
        return

    with open(Q_CLEAN_PATH, "r", encoding="utf-8") as f:
        qs = json.load(f)

    visual_topics = {
        "Non-Verbal Pattern", "Venn Diagrams", "Non-Verbal Series",
        "Spatial Ability", "Non-Verbal Classification", "Non-Verbal Analogy",
        "Dot Situation", "Completion of Figure", "Embedded Figures", 
        "Figure Matrix", "Cube and Dice", "Cubes and Dice", "Mirror Image", "Water Image",
        "Figural Visuals"
    }

    linked_count = 0

    for q in qs:
        # Check if it needs an image or is missing one
        is_visual = q.get("topic") in visual_topics or q.get("section") == "Reasoning"
        has_fig = q.get("has_figure", False)
        
        # We only really care if has_fig is False or it has an empty image_path
        if not has_fig or not q.get("image_path"):
            fname = q.get("file_name", "")
            q_num = q.get("question_number")
            
            if fname and q_num:
                folder_name = str(fname).replace(".pdf", "")
                folder_path = IMG_ROOT / folder_name
                
                if folder_path.exists():
                    # Search for *q{num}*.png
                    pattern1 = f"*q{q_num}_*.png"
                    pattern2 = f"*q{q_num}.png"
                    
                    matches = glob.glob(str(folder_path / pattern1)) + glob.glob(str(folder_path / pattern2))
                    
                    if matches:
                        # Sort to ensure stable ordering
                        matches = sorted(list(set(matches)))
                        # Convert to relative path
                        rel_paths = [Path(m).relative_to(ROOT).as_posix() for m in matches]
                        
                        q["has_figure"] = True
                        q["image_path"] = rel_paths
                        linked_count += 1
                        print(f"Linked {len(rel_paths)} image(s) for {folder_name} Q{q_num}")

    if linked_count > 0:
        with open(Q_CLEAN_PATH, "w", encoding="utf-8") as f:
            json.dump(qs, f, indent=2, ensure_ascii=False)
        print(f"\nSUCCESS: Mapped {linked_count} previously missing images to Q_clean.json!")
    else:
        print("\nNo new images needed to be mapped.")

if __name__ == "__main__":
    main()
