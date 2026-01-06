"""
Simple runner to generate AFCAT questions using local Ollama (llama3).
This script uses `models.question_generator.AFCATQuestionGenerator`.
Run with module mode from project root:
    python -m scripts.generate_ai_questions
"""

import sys
import json
import time
from pathlib import Path

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

# Ensure project root is importable (adds parent directory to path)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from data.question_bank import QuestionBankDB
    from models.question_generator import AFCATQuestionGenerator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you run this from the project root using: python -m scripts.generate_ai_questions")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("output/predictions_2026")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the structure of the Mock Test
SECTIONS = {
    "Verbal Ability": [
        ("Synonyms", 4), ("Antonyms", 4), ("Reading Comprehension", 5),
        ("Cloze Test", 4), ("Error Detection", 5), ("Sentence Completion", 4),
        ("Idioms & Phrases", 4)
    ],
    "General Awareness": [
        ("Current Affairs 2024-2025", 6), ("Defence & Indian Air Force", 5),
        ("Modern Indian History", 4), ("Indian Geography", 4), ("Science & Technology", 3),
        ("Sports 2024-2025", 3)
    ],
    "Numerical Ability": [
        ("Percentage", 4), ("Profit & Loss", 3), ("Time & Work", 3),
        ("Speed Distance Time", 3), ("Ratio & Proportion", 3), ("Average", 2),
        ("Simple & Compound Interest", 2)
    ],
    "Reasoning": [
        ("Coding-Decoding", 5), ("Analogy", 4), ("Number Series", 4),
        ("Odd One Out", 4), ("Logical Reasoning", 4), ("Venn Diagrams", 4)
    ]
}

def normalize_topic_name(name: str) -> str:
    """Basic normalization to pass to generator."""
    return name.replace(" & ", "_").replace(" ", "_").lower()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("🤖 AFCAT 2026 AI QUESTION GENERATOR")
    print("   Using local Ollama (llama3) via models.question_generator")
    print("=" * 70)

    # Initialize Database and Generator
    try:
        db = QuestionBankDB()
        # Note: Ensure Ollama is running (`ollama serve`) before running this script
        gen = AFCATQuestionGenerator(db, ollama_model="llama3", ollama_url="http://localhost:11434")
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return

    # ---------------------------------------------------------
    # PART 1: Generate Full Mock Test
    # ---------------------------------------------------------
    all_questions = []
    print("\n[INFO] Generating full mock test. This may take several minutes...")
    
    for section, topics in SECTIONS.items():
        print(f"\n📚 {section}...")
        for topic, count in topics:
            print(f"   📝 {topic} ({count}q)...", end=" ", flush=True)
            try:
                # Call the generator
                res = gen.generate_questions(
                    topic=normalize_topic_name(topic), 
                    difficulty="medium", 
                    count=count, 
                    use_rag=True
                )
                
                # Handle response
                if hasattr(res, 'questions'):
                    q_count = len(res.questions)
                    print(f"✅ {q_count}")
                    for q in res.questions:
                        # Add section metadata to the question object if missing
                        q_dict = q.to_dict()
                        q_dict['section'] = section
                        all_questions.append(q_dict)
                else:
                    print("❌ (No questions returned)")
            except Exception as e:
                print(f"⚠️ Error: {e}")
            
            # small sleep to prevent overwhelming the local LLM
            time.sleep(0.5)

    # Save Mock Test Output
    mock_output = {
        "metadata": {
            "title": "AFCAT 2026 AI-Generated Mock Test",
            "total_questions": len(all_questions),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_ai_generated": True
        },
        "all_questions": all_questions
    }

    mock_path = OUTPUT_DIR / "ai_mock_test_2026.json"
    with open(mock_path, "w", encoding="utf-8") as f:
        json.dump(mock_output, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Mock test generation complete. Saved {len(all_questions)} questions to {mock_path}")

    # ---------------------------------------------------------
    # PART 2: Generate Predicted Questions (Rising Topics)
    # ---------------------------------------------------------
    rising = [
        ("logical_reasoning", "reasoning", 8),
        ("defence_and_indian_air_force", "general_awareness", 8),
        ("sentence_completion", "verbal_ability", 6),
        ("number_system", "numerical_ability", 5)
    ]

    predicted = []
    print("\n[INFO] Generating predicted questions for rising topics...")
    
    for topic, section, count in rising:
        print(f"   📝 {topic} ({count}q)...", end=" ", flush=True)
        try:
            res = gen.generate_questions(topic=topic, difficulty="medium", count=count, use_rag=True)
            if hasattr(res, 'questions'):
                predicted.extend([q.to_dict() for q in res.questions])
                print(f"✅ {len(res.questions)}")
            else:
                print("❌")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        time.sleep(0.8)

    # Save Prediction Output
    pred_output = {
        "metadata": {
            "title": "AFCAT 2026 Predicted Questions", 
            "total": len(predicted), 
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "questions": predicted
    }
    
    pred_path = OUTPUT_DIR / "ai_predicted_questions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_output, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Predicted questions complete. Saved {len(predicted)} questions to {pred_path}")

if __name__ == "__main__":
    main()