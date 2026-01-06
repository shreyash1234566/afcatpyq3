"""
scripts/finalize_dashboard.py
Reads all generated JSON files and embeds them into a single data.js file.
This allows the dashboard to work locally without a web server.
"""
import json
import os
from pathlib import Path

# Config
OUTPUT_DIR = Path("output/predictions_2026")
DATA_JS_PATH = OUTPUT_DIR / "data.js"

FILES_TO_EMBED = {
    "window.predictionsData": "afcat_2026_predictions.json",
    "window.studyPlanData": "afcat_2026_study_plan.json",
    "window.mockBlueprintData": "afcat_2026_mock_blueprint.json",
    "window.mockSampleQuestions": "afcat_2026_sample_questions.json",
    # Special handling for AI questions to match dashboard expectations
    "ai_mock": "ai_mock_test_2026.json",
    "ai_pred": "ai_predicted_questions.json"
}

def load_json_safe(path):
    if not path.exists():
        print(f"⚠️ Warning: Missing {path.name}")
        return {} if "test" not in path.name else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading {path.name}: {e}")
        return {}

def main():
    print("="*60)
    print("🛠️  FINALIZING DASHBOARD DATA")
    print("="*60)

    if not OUTPUT_DIR.exists():
        print(f"❌ Error: Output directory {OUTPUT_DIR} does not exist.")
        return

    js_content = ["// Auto-generated data file by finalize_dashboard.py"]
    
    # 1. Load Standard Data
    for js_var, filename in FILES_TO_EMBED.items():
        if js_var in ["ai_mock", "ai_pred"]: continue
        
        file_path = OUTPUT_DIR / filename
        data = load_json_safe(file_path)
        json_str = json.dumps(data, ensure_ascii=False)
        js_content.append(f"{js_var} = {json_str};")
        print(f"✅ Embedded {filename} -> {js_var}")

    # 2. Load AI Data (Special handling for dashboardData structure)
    ai_mock_path = OUTPUT_DIR / FILES_TO_EMBED["ai_mock"]
    ai_pred_path = OUTPUT_DIR / FILES_TO_EMBED["ai_pred"]
    
    ai_mock_data = load_json_safe(ai_mock_path)
    # Handle wrapped structure if present (e.g. {"all_questions": [...]})
    if isinstance(ai_mock_data, dict) and "all_questions" in ai_mock_data:
        ai_mock_data = ai_mock_data["all_questions"]
        
    ai_pred_data = load_json_safe(ai_pred_path)
    if isinstance(ai_pred_data, dict) and "questions" in ai_pred_data:
        ai_pred_data = ai_pred_data["questions"]

    dashboard_data = {
        "aiMockQuestions": ai_mock_data if isinstance(ai_mock_data, list) else [],
        "aiPredictedQuestions": ai_pred_data if isinstance(ai_pred_data, list) else [],
        # Add basic stats if missing
        "pyqCount": 2877, 
        "risingTopics": [] # Will be populated by frontend from predictionsData
    }
    
    js_content.append(f"window.dashboardData = {json.dumps(dashboard_data, ensure_ascii=False)};")
    print(f"✅ Embedded AI Questions -> window.dashboardData")

    # 3. Add Dummy PYQ Data if missing (to prevent crash)
    # In a real scenario, this would load the full Q.json, but that might be too large for JS.
    # We'll add an empty placeholder or a small sample.
    js_content.append("window.pyqData = [];") 
    print("✅ Added placeholder for window.pyqData")

    # 4. Write to data.js
    with open(DATA_JS_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(js_content))

    print("-" * 60)
    print(f"🎉 Success! Dashboard data written to: {DATA_JS_PATH}")
    print("👉 You can now open dashboard.html directly in your browser.")

if __name__ == "__main__":
    main()