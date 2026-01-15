import json
import os
import re
from topic_normalization_map import TOPIC_NORMALIZATION

INPUT_PATH = "data/processed/Q.json"
OUTPUT_PATH = "data/processed/Q_normalized.json"

# --- OPTIMIZATION ---
TOPIC_MAP_LOWER = {k.lower(): v for k, v in TOPIC_NORMALIZATION.items()}

def normalize_topic(raw_topic):
    """Normalizes a topic string using the dictionary."""
    if not isinstance(raw_topic, str) or not raw_topic.strip():
        return "Unknown"

    clean_key = raw_topic.strip()

    if clean_key in TOPIC_NORMALIZATION:
        return TOPIC_NORMALIZATION[clean_key]

    clean_key_lower = clean_key.lower()
    if clean_key_lower in TOPIC_MAP_LOWER:
        return TOPIC_MAP_LOWER[clean_key_lower]

    return clean_key

def extract_year(filename):
    """Extracts year (20xx) from filename string."""
    if not filename:
        return None
    match = re.search(r'(20\d{2})', str(filename))
    return int(match.group(1)) if match else None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: File not found at {INPUT_PATH}")
        return

    print("Loading data...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} questions...")
    
    topic_changes = 0
    year_changes = 0

    # --- STEP 1: PRE-PROCESS YEARS ---
    # We store a temporary year for everyone to allow neighbor checking
    for q in data:
        q['temp_year'] = extract_year(q.get('file_name', ''))

    # --- STEP 2: INFER MISSING YEARS (Your Logic) ---
    for i in range(len(data)):
        # If year is missing
        if data[i]['temp_year'] is None:
            prev_year = data[i-1]['temp_year'] if i > 0 else None
            next_year = data[i+1]['temp_year'] if i < len(data) - 1 else None
            
            # Logic: If upper and lower files have the same year, use it
            if prev_year and next_year and prev_year == next_year:
                inferred_year = prev_year
                data[i]['temp_year'] = inferred_year
                
                # Update the file name to include the year (Renaming step)
                current_name = data[i].get('file_name', 'Unknown_File')
                data[i]['file_name'] = f"{inferred_year}_{current_name}"
                year_changes += 1

    # --- STEP 3: NORMALIZE TOPICS & FINALIZE ---
    for q in data:
        # Finalize Year
        if q.get('temp_year'):
            q['year'] = q['temp_year']
        del q['temp_year'] # Remove temp field

        # Normalize Topic
        orig = q.get("topic_name") or q.get("topic")
        norm = normalize_topic(orig)
        
        if orig != norm:
            topic_changes += 1
            
        if "topic_name" in q:
            q["topic_name"] = norm
        elif "topic" in q:
            q["topic"] = norm
        else:
            q["topic_name"] = norm

    # Save output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Success! Processed data written to {OUTPUT_PATH}")
    print(f"Topics standardized: {topic_changes}")
    print(f"Missing years inferred & files renamed: {year_changes}")

if __name__ == "__main__":
    main()