"""
Update dashboard data.js with AI-generated questions
"""

import json
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output/predictions_2026")
DATA_DIR = Path("data/processed")


def load_json_safe(filepath):
    """Load JSON with error handling."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pyq():
    """Load PYQ with malformed JSON fix."""
    with open(DATA_DIR / "Q.json", 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = []
        decoder = json.JSONDecoder()
        pos = 0
        content = content.strip()
        while pos < len(content):
            try:
                obj, end = decoder.raw_decode(content, pos)
                if isinstance(obj, list):
                    data.extend(obj)
                elif isinstance(obj, dict):
                    data.append(obj)
                pos = end
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
            except json.JSONDecodeError:
                pos += 1
    
    # Fix years
    for q in data:
        if not q.get('year'):
            fn = q.get('file_name', '')
            match = re.search(r'AFCAT[_\s]*(\d{4})', fn, re.IGNORECASE)
            if match:
                q['year'] = int(match.group(1))
    
    return data


def main():
    print("=" * 60)
    print("📦 Updating Dashboard Data with AI Questions")
    print("=" * 60)
    
    # Load all data
    predictions = load_json_safe(OUTPUT_DIR / "afcat_2026_predictions.json")
    ai_mock = load_json_safe(OUTPUT_DIR / "ai_mock_test_2026.json")
    ai_predicted = load_json_safe(OUTPUT_DIR / "ai_predicted_questions.json")
    pyq_data = load_pyq()
    
    print(f"✓ PYQ: {len(pyq_data)} questions")
    print(f"✓ AI Mock: {len(ai_mock['all_questions'])} questions")
    print(f"✓ AI Predicted: {len(ai_predicted['questions'])} questions")
    
    # Create comprehensive data.js
    js_content = f"""// AFCAT 2026 Dashboard Data - WITH AI GENERATED QUESTIONS
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
// AI Questions: {len(ai_mock['all_questions'])} mock + {len(ai_predicted['questions'])} predicted

const predictionsData = {json.dumps(predictions, ensure_ascii=False)};

const pyqData = {json.dumps(pyq_data, ensure_ascii=False)};

const aiMockTest = {json.dumps(ai_mock, ensure_ascii=False)};

const aiPredictedQuestions = {json.dumps(ai_predicted, ensure_ascii=False)};

// For backwards compatibility with dashboard
const mockBlueprintData = {{
    metadata: aiMockTest.metadata,
    all_questions: aiMockTest.all_questions,
    sections: aiMockTest.sections
}};

const aiSampleQuestions = {{
    sections: {{}}
}};

// Organize predicted questions by section
aiPredictedQuestions.questions.forEach(q => {{
    const section = q.section || 'General';
    if (!aiSampleQuestions.sections[section]) {{
        aiSampleQuestions.sections[section] = [];
    }}
    aiSampleQuestions.sections[section].push(q);
}});

// Study plan placeholder
const studyPlanData = {{
    metadata: {{
        total_days: 20,
        hours_per_day: 6,
        total_hours: 120
    }},
    daily_schedule: []
}};

// Generate 20-day study plan
const phases = [
    {{name: 'Concept Building', days: [1,2,3,4,5,6,7,8], color: '#3b82f6'}},
    {{name: 'Advanced Practice', days: [9,10,11,12,13,14], color: '#10b981'}},
    {{name: 'Mock Tests', days: [15,16,17], color: '#f59e0b'}},
    {{name: 'Final Revision', days: [18,19,20], color: '#ef4444'}}
];

const sections = ['Verbal Ability', 'General Awareness', 'Numerical Ability', 'Reasoning'];

for (let day = 1; day <= 20; day++) {{
    let phase = phases.find(p => p.days.includes(day));
    let sectionIdx = (day - 1) % 4;
    
    studyPlanData.daily_schedule.push({{
        day: day,
        phase: phase.name,
        phase_description: `Focus on ${{sections[sectionIdx]}} with practice questions`,
        total_hours: 6,
        sessions: [
            {{
                time: '6:00 AM - 8:30 AM',
                duration: '2.5 hours',
                section: sections[sectionIdx],
                topics: ['High Priority Topics'],
                activity: day <= 8 ? 'Theory + Concepts' : day <= 14 ? 'Practice Problems' : day <= 17 ? 'Mock Test' : 'Revision',
                target: day <= 8 ? 'Complete theory, solve 25 questions' : day <= 14 ? 'Timed practice, 30 questions' : day <= 17 ? 'Full mock analysis' : 'Quick revision'
            }},
            {{
                time: '10:00 AM - 12:00 PM',
                duration: '2 hours',
                section: sections[(sectionIdx + 1) % 4],
                topics: ['Secondary Topics'],
                activity: 'Practice',
                target: 'Solve 20 questions'
            }},
            {{
                time: '4:00 PM - 5:30 PM',
                duration: '1.5 hours',
                section: 'Mixed Practice',
                topics: ['PYQ Practice', 'Weak Areas'],
                activity: 'PYQ Solving',
                target: 'Solve 25 PYQ questions'
            }}
        ]
    }});
}}

console.log('✅ Dashboard data loaded');
console.log('   PYQ:', pyqData.length, 'questions');
console.log('   AI Mock:', aiMockTest.all_questions.length, 'NEW questions');
console.log('   AI Predicted:', aiPredictedQuestions.questions.length, 'questions');
console.log('   Rising Topics:', predictionsData.rising_topics?.length || 0);
"""
    
    # Save
    with open(OUTPUT_DIR / "data.js", 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"\n✅ data.js created: {len(js_content):,} bytes")
    print(f"📁 Saved to: {OUTPUT_DIR / 'data.js'}")


if __name__ == "__main__":
    main()
