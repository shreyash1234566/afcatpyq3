"""
Update Dashboard Data Files
============================
Regenerates temp_data.js and dashboard/data.js to reflect
the normalized topic names from Q.json.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(r"e:\afcatpyq3")
Q_JSON_PATH = BASE_DIR / "data" / "processed" / "Q.json"
TEMP_DATA_PATH = BASE_DIR / "temp_data.js"
DASHBOARD_DATA_PATH = BASE_DIR / "dashboard" / "data.js"

# Topic name normalization mapping for temp_data.js (old fragmented → new canonical)
# This handles the topic_name fields in predictions, study plan, etc.
TEMP_DATA_TOPIC_RENAMES = {
    # Reasoning renames
    "Non-Verbal Dot Situation": "Dot Situation",
    "Order & Ranking": "Logical Reasoning",
    "Verbal Logic": "Logical Reasoning",
    "Verbal Puzzle": "Logical Reasoning",
    "Character Puzzle": "Logical Reasoning",
    "Puzzle": "Logical Reasoning",
    "Seating Arrangement": "Logical Reasoning",
    "Ranking": "Logical Reasoning",
    "Ranking & Ordering": "Logical Reasoning",
    "Mathematical Operations": "Logical Reasoning",
    "Logical Arrangement": "Logical Reasoning",
    "Logical Reasoning (Verbal)": "Logical Reasoning",
    "Coding and Decoding": "Coding-Decoding",
    "Coding Decoding": "Coding-Decoding",
    "Verbal Coding": "Coding-Decoding",
    "Verbal Coding-Decoding": "Coding-Decoding",
    "Venn Diagram": "Venn Diagrams",
    "Venn Diagram (Verbal)": "Venn Diagrams",
    "Verbal Venn": "Venn Diagrams",
    "Non-Verbal Venn Diagrams": "Venn Diagrams",
    "Non-Verbal Venn Diagram": "Venn Diagrams",
    "Analogy (Verbal)": "Verbal Analogy",
    "Analogy": "Verbal Analogy",
    "Number Analogy": "Verbal Analogy",
    "Classification (Verbal)": "Verbal Classification",
    "Classification": "Verbal Classification",
    "Classification (Jumbled Words)": "Verbal Classification",
    "Classification (Letter)": "Verbal Classification",
    "Classification (Number)": "Verbal Classification",
    "Verbal Series": "Number/Letter Series",
    "Number Series": "Number/Letter Series",
    "Letter Series": "Number/Letter Series",
    "Series": "Number/Letter Series",
    "Non-Verbal Pattern Completion": "Non-Verbal Pattern",
    "Non-Verbal Puzzle": "Non-Verbal Pattern",
    "Pattern": "Non-Verbal Pattern",
    "Pattern Completion": "Non-Verbal Pattern",
    "Completion of Pattern": "Non-Verbal Pattern",
    "Pattern Recognition": "Non-Verbal Pattern",
    "Non-Verbal Figure Series": "Non-Verbal Series",
    "Non-Verbal Figure Formation": "Non-Verbal Pattern",
    "Non-Verbal Embedded Figures": "Embedded Figures",
    "Non-Verbal Embedded": "Embedded Figures",
    "Non-Verbal Spatial (Embedded)": "Embedded Figures",
    "Non-Verbal Dot Situation": "Dot Situation",
    "Non-Verbal Orientation": "Spatial Ability",
    "Non-Verbal Spatial": "Spatial Ability",
    "Non-Verbal Spatial Ability": "Spatial Ability",
    "Non-Verbal Spatial Reasoning": "Spatial Ability",
    "Spatial Reasoning": "Spatial Ability",
    "Spatial Orientation": "Direction Sense",
    "Direction and Distance": "Direction Sense",
    "Direction & Distance": "Direction Sense",
    "Direction Sense Test": "Direction Sense",
    
    # Verbal Ability renames
    "Spotting Errors": "Error Detection",
    "Error Spotting": "Error Detection",
    "Comprehension": "Reading Comprehension",
    "Idioms and Phrases": "Idioms & Phrases",
    "Idioms": "Idioms & Phrases",
    "Para Jumbles": "Sentence Rearrangement",
    "Spelling Test": "Spelling",
    "Grammar (Spellings)": "Spelling",
    "Tenses": "Grammar",
    "Tense and Grammar": "Grammar",
    "Grammar (Voice)": "Grammar",
    "Prepositions": "Grammar",
}


def normalize_topic_in_text(text):
    """Replace old topic names with new ones in any text string."""
    for old, new in TEMP_DATA_TOPIC_RENAMES.items():
        text = text.replace(old, new)
    return text


def update_temp_data():
    """Update temp_data.js with normalized topic names."""
    print("\n  Updating temp_data.js...")
    
    with open(TEMP_DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Strip the JS wrapper to get pure JSON
    json_str = content.strip()
    if json_str.startswith("const dashboardData = "):
        json_str = json_str[len("const dashboardData = "):]
    if json_str.endswith(";"):
        json_str = json_str[:-1]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"    [!] Failed to parse temp_data.js as JSON: {e}")
        # Fall back to text replacement
        new_content = normalize_topic_in_text(content)
        with open(TEMP_DATA_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("    [OK] Applied text-based topic name replacements")
        return
    
    # Deep walk the data structure and rename topics
    renames_done = [0]
    
    def walk_and_rename(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("topic_name", "topic") and isinstance(value, str):
                    if value in TEMP_DATA_TOPIC_RENAMES:
                        obj[key] = TEMP_DATA_TOPIC_RENAMES[value]
                        renames_done[0] += 1
                elif key == "topics" and isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str) and item in TEMP_DATA_TOPIC_RENAMES:
                            value[i] = TEMP_DATA_TOPIC_RENAMES[item]
                            renames_done[0] += 1
                else:
                    walk_and_rename(value)
        elif isinstance(obj, list):
            for item in obj:
                walk_and_rename(item)
    
    walk_and_rename(data)
    
    # Also rename topic codes
    # Update topic_code mappings where needed
    code_renames = {
        "RM_MISC": "RM_NV_DOT",  # Non-Verbal Dot Situation was RM_MISC
        "RM_VR_ORDER": "RM_VR_LOG",  # Order & Ranking merged into Logical Reasoning
    }
    
    def walk_and_rename_codes(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "topic_code" and isinstance(value, str) and value in code_renames:
                    obj[key] = code_renames[value]
                    renames_done[0] += 1
                else:
                    walk_and_rename_codes(value)
        elif isinstance(obj, list):
            for item in obj:
                walk_and_rename_codes(item)
    
    walk_and_rename_codes(data)
    
    # Write back
    new_content = "const dashboardData = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n"
    with open(TEMP_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"    [OK] Renamed {renames_done[0]} topic references in temp_data.js")


def update_dashboard_data():
    """Update dashboard/data.js with normalized topic names."""
    print("\n  Updating dashboard/data.js...")
    
    with open(DASHBOARD_DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    json_str = content.strip()
    if json_str.startswith("const dashboardData = "):
        json_str = json_str[len("const dashboardData = "):]
    if json_str.endswith(";"):
        json_str = json_str[:-1]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"    [!] Failed to parse dashboard/data.js as JSON: {e}")
        new_content = normalize_topic_in_text(content)
        with open(DASHBOARD_DATA_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("    [OK] Applied text-based topic name replacements")
        return
    
    renames_done = [0]
    
    def walk_and_rename(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("topic_name", "topic") and isinstance(value, str):
                    if value in TEMP_DATA_TOPIC_RENAMES:
                        obj[key] = TEMP_DATA_TOPIC_RENAMES[value]
                        renames_done[0] += 1
                elif key == "topics" and isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str) and item in TEMP_DATA_TOPIC_RENAMES:
                            value[i] = TEMP_DATA_TOPIC_RENAMES[item]
                            renames_done[0] += 1
                else:
                    walk_and_rename(value)
        elif isinstance(obj, list):
            for item in obj:
                walk_and_rename(item)
    
    walk_and_rename(data)
    
    new_content = "const dashboardData = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n"
    with open(DASHBOARD_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"    [OK] Renamed {renames_done[0]} topic references in dashboard/data.js")


def print_q_json_summary():
    """Print current Q.json topic distribution."""
    with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("\n" + "=" * 70)
    print("  CURRENT Q.JSON TOPIC SUMMARY")
    print("=" * 70)
    
    for section_name in ["Reasoning", "Verbal Ability", "General Awareness", "Numerical Ability"]:
        topics = Counter(q["topic"] for q in data if q.get("section") == section_name)
        if topics:
            print(f"\n  --- {section_name} ({sum(topics.values())} questions, {len(topics)} topics) ---")
            for topic, count in topics.most_common():
                print(f"    {count:4d}  {topic}")


def main():
    print("=" * 70)
    print("  DASHBOARD DATA UPDATER")
    print("=" * 70)
    
    update_temp_data()
    update_dashboard_data()
    print_q_json_summary()
    
    print("\n  [OK] All dashboard data files updated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
