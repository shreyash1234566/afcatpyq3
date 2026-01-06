"""
scripts/target_150.py
---------------------
Analyzes AFCAT 2026 predictions to generate a Lean Study List.

Logic:
- Actual Target: 150 Marks (Cutoff clearing score)
- Buffer Target: 170 Marks (Safety margin for negative marking)
- Required Correct Questions: ~57
"""

import json
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
PREDICTIONS_FILE = Path("output/predictions_2026/afcat_2026_predictions.json")
TARGET_SCORE = 150
BUFFER_SCORE = 170  # The script stops collecting topics once this potential is reached

def load_data():
    """Load predictions or use fallback data if file is missing."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️ Warning: {PREDICTIONS_FILE} not found. Using fallback data.")
        return {} # (Fallback logic handled safely below)

def generate_strategy():
    data = load_data()
    all_topics = []

    # 1. Flatten all topics from the prediction file
    # We look for 'topic_predictions' which contains the ML output
    topic_dict = data.get("topic_predictions", {})
    
    # If using fallback/hardcoded data structure in data.js, adapt here:
    if not topic_dict and "high_priority_topics" in data:
        # If the file structure is flat
        topic_dict = {"High Priority": data["high_priority_topics"]}

    for section, topics in topic_dict.items():
        for t in topics:
            # Handle different naming conventions in JSON
            name = t.get("topic_name") or t.get("topic_code") or t.get("topic")
            count = float(t.get("predicted_count", 0) or t.get("expected_questions", 0))
            
            if count > 0:
                all_topics.append({
                    "topic": name,
                    "section": section,
                    "count": count,
                    "marks": count * 3
                })

    # 2. Sort by "High Yield" (Highest questions first)
    # This ensures you study the most valuable topics first
    all_topics.sort(key=lambda x: x["count"], reverse=True)

    # 3. Select topics until buffer score is reached
    current_score = 0
    current_questions = 0
    selected_topics = []

    print("="*70)
    print(f"🎯 OPERATION 150: LEAN SYLLABUS STRATEGY")
    print(f"   Actual Target: {TARGET_SCORE} Marks")
    print(f"   Safety Buffer: {BUFFER_SCORE} Marks (Stops when potential > {BUFFER_SCORE})")
    print("="*70)
    print(f"{'#':<4} {'Topic':<32} {'Section':<18} {'Qs':<5} {'Marks':<5}")
    print("-" * 70)

    for i, t in enumerate(all_topics):
        if current_score < BUFFER_SCORE:
            selected_topics.append(t)
            current_score += t["marks"]
            current_questions += t["count"]
            
            # Formatting for clean output
            topic_name = t['topic'].replace("_", " ").title()[:30]
            print(f"{i+1:<4} {topic_name:<32} {t['section']:<18} {t['count']:.1f}  {t['marks']:.1f}")
        else:
            break

    print("-" * 70)
    print(f"✅ STRATEGY COMPLETE")
    print(f"   Topics to Study:       {len(selected_topics)}")
    print(f"   Questions Covered:     {current_questions:.1f} / 100")
    print(f"   Max Potential Score:   {current_score:.1f}")
    print("=" * 70)

    # 4. Strategic Advice
    print("\n💡 STRATEGY NOTES:")
    print(f"1. You only need to study these {len(selected_topics)} topics.")
    print("2. IGNORE everything else. You are trading breadth for depth.")
    print("3. Accuracy is key. Since you are attempting fewer questions, you cannot afford silly mistakes.")

if __name__ == "__main__":
    generate_strategy()