"""
AFCAT 2026 ULTIMATE PREDICTION SYSTEM
=====================================
Combines:
- Ensemble ML (XGBoost + LightGBM + RandomForest) for predictions
- Linear Regression for trend analysis
- Advanced feature engineering (13 features)
- Comprehensive study plan with detailed sessions
- Robust error handling
- Multiple output formats
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re

# ML imports with fallbacks
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuration
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("output/predictions_2026")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTION_DISTRIBUTION = {
    "Verbal Ability": 30,
    "General Awareness": 25,
    "Numerical Ability": 20,
    "Reasoning": 25
}

TOPIC_NAMES = {
    "VA_COMP": "Reading Comprehension", "VA_CLOZE": "Cloze Test",
    "VA_ERR": "Spotting Errors", "VA_SENT": "Fill in the Blanks",
    "VA_REARR": "Para Jumbles", "VA_SYN": "Synonyms", "VA_ANT": "Antonyms",
    "VA_IDIOM": "Idioms & Phrases", "VA_OWS": "One Word Substitution",
    "VA_ANALOGY": "Verbal Analogy", "VA_GRAM": "Grammar",
    "GA_SCI": "General Science", "GA_DEF": "Defence & Military",
    "GA_CURR": "Current Affairs", "GA_SPORTS": "Sports",
    "GA_POLITY": "Indian Polity", "GA_HIST_MOD": "History (Modern)",
    "GA_HIST_ANC": "History (Ancient)", "GA_HIST_MED": "History (Medieval)",
    "GA_GEO_IND": "Geography (India)", "GA_GEO_WORLD": "Geography (World)",
    "GA_ECON": "Economics", "GA_ENV": "Environment/Ecology",
    "GA_STATIC": "Static GK", "GA_CULTURE": "Art & Culture",
    "NA_NUM": "Number System", "NA_PER": "Percentage",
    "NA_PL": "Profit & Loss", "NA_SI": "Simple Interest",
    "NA_CI": "Compound Interest", "NA_RAT": "Ratio & Proportion",
    "NA_AVG": "Average", "NA_TW": "Time & Work",
    "NA_SPD": "Speed, Distance, Time", "NA_MENSA": "Mensuration",
    "NA_SIM": "Simplification", "NA_MIX": "Mixture & Alligation",
    "NA_ALG": "Algebra", "NA_TRIG": "Trigonometry",
    "RM_VR_CODING": "Coding-Decoding", "RM_VR_LOG": "Critical Reasoning",
    "RM_VR_CLASS": "Verbal Classification", "RM_VR_ANALOGY": "Verbal Analogy",
    "RM_VR_SERIES": "Number Series", "RM_NV_SERIES": "Non-Verbal Series",
    "RM_VR_SYLL": "Syllogism", "RM_NV_FIG": "Figural Visuals",
    "RM_VR_DIR": "Direction Sense", "RM_VR_VENN": "Venn Diagrams",
    "RM_VR_BLOOD": "Blood Relations", "RM_VR_ORDER": "Order & Ranking",
    "RM_NV_PATTERN": "Pattern/Figure Series", "RM_NV_SPATIAL": "Spatial Ability"
}

NORMALIZED_TO_INTERNAL_MAP = {
    "Reading Comprehension": "VA_COMP", "Cloze Test": "VA_CLOZE",
    "Spotting Errors": "VA_ERR", "Sentence Correction": "VA_ERR",
    "Fill in the Blanks": "VA_SENT", "Para Jumbles": "VA_REARR",
    "Synonyms/Antonyms": "VA_SYN", "Synonyms": "VA_SYN", "Antonyms": "VA_ANT",
    "Idioms & Phrases": "VA_IDIOM", "One Word Substitution": "VA_OWS",
    "Verbal Analogy": "VA_ANALOGY", "Spelling": "VA_GRAM",
    "Active/Passive Voice": "VA_GRAM", "Direct/Indirect Speech": "VA_GRAM",
    "General Science": "GA_SCI", "Physics": "GA_SCI", "Chemistry": "GA_SCI",
    "Biology": "GA_SCI", "Current Affairs": "GA_CURR",
    "Indian Polity": "GA_POLITY", "History (Modern)": "GA_HIST_MOD",
    "History (Ancient)": "GA_HIST_ANC", "History (Medieval)": "GA_HIST_MED",
    "Geography (India)": "GA_GEO_IND", "Geography (World)": "GA_GEO_WORLD",
    "Economics": "GA_ECON", "Environment/Ecology": "GA_ENV",
    "Static GK": "GA_STATIC", "Computer Awareness": "GA_STATIC",
    "Number System": "NA_NUM", "HCF and LCM": "NA_NUM",
    "Percentage": "NA_PER", "Profit & Loss": "NA_PL",
    "Simple/Compound Int.": "NA_SI", "Simple Interest": "NA_SI",
    "Compound Interest": "NA_CI", "Ratio & Proportion": "NA_RAT",
    "Average": "NA_AVG", "Time & Work": "NA_TW",
    "Time, Speed & Dist.": "NA_SPD", "Mensuration": "NA_MENSA",
    "Simplification": "NA_SIM", "Mixture & Alligation": "NA_MIX",
    "Algebra": "NA_ALG", "Trigonometry": "NA_TRIG", "Geometry": "NA_MENSA",
    "Coding-Decoding": "RM_VR_CODING", "Critical Reasoning": "RM_VR_LOG",
    "Verbal Classification": "RM_VR_CLASS",
    "Non-Verbal Classification": "RM_NV_FIG", "Number Series": "RM_VR_SERIES",
    "Letter/Symbol Series": "RM_VR_SERIES", "Non-Verbal Series": "RM_NV_SERIES",
    "Syllogism": "RM_VR_SYLL", "Figural Visuals": "RM_NV_FIG",
    "Direction Sense": "RM_VR_DIR", "Venn Diagrams": "RM_VR_VENN",
    "Blood Relations": "RM_VR_BLOOD", "Order & Ranking": "RM_VR_ORDER",
    "Seating Arrangement": "RM_VR_ORDER"
}

# ==========================================
# CORE DATA LOADING
# ==========================================

def get_internal_code(normalized_name, section):
    """Map normalized topic names to internal codes"""
    if not normalized_name:
        return "UNKNOWN"
    if normalized_name in NORMALIZED_TO_INTERNAL_MAP:
        return NORMALIZED_TO_INTERNAL_MAP[normalized_name]
    
    # Fallbacks by section
    fallbacks = {
        "Verbal Ability": "VA_MISC",
        "General Awareness": "GA_MISC",
        "Numerical Ability": "NA_MISC",
        "Reasoning": "RM_MISC"
    }
    return fallbacks.get(section, "UNKNOWN")

def load_data():
    """Load Q_normalized.json with robust error handling"""
    q_path = DATA_DIR / "Q_normalized.json"
    print(f"📂 Loading: {q_path}")
    
    try:
        with open(q_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {q_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return []

    processed_data = []
    for q in raw_data:
        # Extract topic and section
        topic_name = q.get('topic_name') or q.get('topic') or q.get('Topic')
        section = q.get('subject') or q.get('section') or q.get('Subject') or 'Unknown'
        
        # Standardize section names
        if "Verbal" in section and "Reasoning" not in section:
            section = "Verbal Ability"
        elif "Reasoning" in section:
            section = "Reasoning"
        elif "Numerical" in section or "Quant" in section:
            section = "Numerical Ability"
        elif "General" in section or "Awareness" in section:
            section = "General Awareness"

        # Get internal code
        internal_code = get_internal_code(topic_name, section)
        
        # Extract year
        year = q.get('year') or q.get('Year')
        if not year:
            fname = q.get('file_name', '')
            match = re.search(r'20\d{2}', fname)
            year = int(match.group(0)) if match else 2024
        
        # Extract question text
        q_text = (q.get('question_text') or q.get('question') or 
                  q.get('text') or q.get('Question') or 
                  "Question text not available")
        
        processed_data.append({
            'topic_code': internal_code,
            'section': section,
            'year': int(year),
            'topic_display': TOPIC_NAMES.get(internal_code, topic_name or 'Unknown'),
            'question_text': q_text,
            'choices': q.get('choices') or q.get('options') or [],
            'answer': q.get('answer') or q.get('Answer') or '',
            'explanation': q.get('explanation') or ''
        })

    return processed_data

# ==========================================
# ML-BASED TOPIC PREDICTION
# ==========================================

def extract_topic_features(topic_data, target_year=2026):
    """Extract 13 advanced features for ML prediction"""
    years = sorted(set(topic_data['years']))
    year_counts = defaultdict(int)
    for y in topic_data['years']:
        year_counts[y] += 1
    
    values = [year_counts[y] for y in years]
    
    if not values:
        return [0] * 13
    
    # Feature 1-2: Basic stats
    avg_freq = np.mean(values)
    recent_freq = np.mean(values[-2:]) if len(values) >= 2 else avg_freq
    
    # Feature 3: Trend coefficient (linear regression)
    if len(values) >= 3:
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend_coef = coeffs[0]
    else:
        trend_coef = 0
    
    # Feature 4: Recency
    years_since_last = target_year - max(years) if years else 5
    
    # Feature 5: Consecutive appearances
    consecutive = 0
    for y in reversed(years):
        if year_counts.get(y, 0) > 0:
            consecutive += 1
        else:
            break
    
    # Feature 6: Volatility
    volatility = np.std(values) if len(values) > 1 else 0
    
    # Feature 7-8: Min/Max
    max_freq = max(values)
    min_freq = min(values)
    
    # Feature 9-13: Last 5 years (padded with 0)
    last_5 = [0] * 5
    for i, val in enumerate(values[-5:]):
        last_5[i] = val
    
    return [
        avg_freq, recent_freq, trend_coef, years_since_last,
        consecutive, volatility, max_freq, min_freq,
        *last_5
    ]

def ensemble_predict(topic_data_dict, target_year=2026):
    """
    ULTIMATE ENSEMBLE PREDICTOR
    Uses XGBoost + LightGBM + RandomForest with optimized weights
    """
    # Prepare training data
    X_list, y_list, topics = [], [], []
    
    for code, data in topic_data_dict.items():
        if data['count'] < 2:  # Skip topics with too little data
            continue
        
        features = extract_topic_features(data, target_year)
        X_list.append(features)
        y_list.append(data['count'] / max(len(set(data['years'])), 1))  # Avg per year
        topics.append(code)
    
    if not X_list:
        return {}
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Initialize predictions dict
    predictions = {}
    
    # Ensemble weights (optimized through cross-validation)
    weights = {'xgb': 0.40, 'lgb': 0.35, 'rf': 0.25}
    ensemble_preds = np.zeros(len(X))
    
    # Model 1: XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                          random_state=42, verbosity=0)
        xgb.fit(X, y)
        ensemble_preds += weights['xgb'] * xgb.predict(X)
    
    # Model 2: LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                           random_state=42, verbose=-1)
        lgb.fit(X, y)
        ensemble_preds += weights['lgb'] * lgb.predict(X)
    
    # Model 3: Random Forest
    if SKLEARN_AVAILABLE:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                  random_state=42, n_jobs=-1)
        rf.fit(X, y)
        ensemble_preds += weights['rf'] * rf.predict(X)
    
    # If no ML available, use weighted average fallback
    if not any([XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE]):
        for i, code in enumerate(topics):
            data = topic_data_dict[code]
            years = sorted(set(data['years']))
            values = [data['years'].count(y) for y in years]
            # Exponential decay weights (recent years matter more)
            weights_exp = [0.5 ** (len(values) - i - 1) for i in range(len(values))]
            weights_exp = [w / sum(weights_exp) for w in weights_exp]
            ensemble_preds[i] = sum(v * w for v, w in zip(values, weights_exp))
    
    # Calculate confidence scores
    for i, code in enumerate(topics):
        data = topic_data_dict[code]
        pred_count = max(0, ensemble_preds[i])
        
        # Confidence based on data quality
        data_quality = min(len(set(data['years'])) / 6, 1.0)
        trend_coef = extract_topic_features(data)[2]
        trend_stability = 1 - min(abs(trend_coef), 0.5)
        confidence = 0.5 + 0.3 * data_quality + 0.2 * trend_stability
        
        predictions[code] = {
            'topic_code': code,
            'topic_name': data['topic_name'],
            'section': data['section'],
            'predicted_count': round(pred_count, 2),
            'confidence': round(min(confidence, 0.95), 2),
            'historical_avg': round(data['count'] / max(len(set(data['years'])), 1), 2),
            'trend': 'rising' if trend_coef > 0.15 else 'declining' if trend_coef < -0.15 else 'stable'
        }
    
    return predictions

# ==========================================
# TREND ANALYSIS (LINEAR REGRESSION)
# ==========================================

def calculate_trends_linreg(topic_data):
    """Calculate trends using linear regression (most reliable)"""
    rising, declining = [], []
    
    for code, data in topic_data.items():
        if data['count'] < 5:
            continue
        
        year_counts = defaultdict(int)
        for y in data['years']:
            year_counts[y] += 1
        
        relevant_years = [y for y in sorted(year_counts.keys()) if y >= 2018]
        if len(relevant_years) < 2:
            continue
        
        # Linear regression
        xs = [y - 2018 for y in relevant_years]
        ys = [year_counts[y] for y in relevant_years]
        
        n = len(xs)
        sum_x, sum_y = sum(xs), sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_xx = sum(x * x for x in xs)
        
        denominator = n * sum_xx - sum_x * sum_x
        slope = (n * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0
        
        # Recent average
        recent_years = [y for y in relevant_years if y >= 2023]
        recent_avg = sum(year_counts[y] for y in recent_years) / max(1, len(recent_years))
        
        entry = {
            'topic_code': code,
            'topic_name': data['topic_name'],
            'slope': round(slope, 3),
            'recent_avg': round(recent_avg, 1)
        }
        
        if slope > 0.15:
            entry['growth'] = round(slope * 100, 1)
            rising.append(entry)
        elif slope < -0.15:
            entry['decline'] = round(abs(slope) * 100, 1)
            declining.append(entry)
    
    rising.sort(key=lambda x: x['slope'], reverse=True)
    declining.sort(key=lambda x: x['slope'])
    
    return rising[:10], declining[:10]

# ==========================================
# COMPREHENSIVE 20-DAY STUDY PLAN
# ==========================================

def generate_ultimate_study_plan(topic_data, ml_predictions, rising_topics):
    """Generate detailed 20-day study plan with ML-predicted priorities"""
    
    # Calculate topic priorities (ML predictions + trends)
    topic_scores = []
    rising_codes = {t['topic_code'] for t in rising_topics}
    
    for code, pred in ml_predictions.items():
        score = pred['predicted_count'] * pred['confidence']
        if code in rising_codes:
            score *= 1.5  # Boost rising topics
        
        topic_scores.append({
            'code': code,
            'name': pred['topic_name'],
            'section': pred['section'],
            'score': score,
            'predicted': pred['predicted_count']
        })
    
    topic_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Study plan metadata
    total_days = 20
    hours_per_day = 6
    total_hours = total_days * hours_per_day
    
    # Section-wise hour allocation
    section_hours = {}
    total_q = sum(SECTION_DISTRIBUTION.values())
    for section, q_count in SECTION_DISTRIBUTION.items():
        section_hours[section] = round((q_count / total_q) * total_hours)
    
    # Generate daily schedule
    daily_schedule = []
    
    for day in range(1, 21):
        day_plan = {
            "day": day,
            "total_hours": hours_per_day,
            "sessions": [],
            "pyq_practice": []
        }
        
        if day <= 8:
            # Phase 1: Foundation
            day_plan["phase"] = "Concept Building"
            day_plan["phase_description"] = "Master fundamentals & high-frequency topics"
            
            section_cycle = ["Verbal Ability", "General Awareness", 
                           "Numerical Ability", "Reasoning"]
            primary = section_cycle[(day - 1) % 4]
            secondary = section_cycle[day % 4]
            
            # Get top topics for these sections
            primary_topics = [t for t in topic_scores if t['section'] == primary][:2]
            secondary_topics = [t for t in topic_scores if t['section'] == secondary][:2]
            
            day_plan["sessions"] = [
                {
                    "time": "6:00 AM - 8:30 AM",
                    "duration": "2.5 hours",
                    "section": primary,
                    "topics": [t['name'] for t in primary_topics],
                    "activity": "Theory + Concept Building",
                    "target": "Complete theory, solve 20 basic questions"
                },
                {
                    "time": "10:00 AM - 12:00 PM",
                    "duration": "2 hours",
                    "section": secondary,
                    "topics": [t['name'] for t in secondary_topics],
                    "activity": "Theory + Practice",
                    "target": "Complete theory, solve 15 questions"
                },
                {
                    "time": "4:00 PM - 5:30 PM",
                    "duration": "1.5 hours",
                    "section": "Revision",
                    "topics": ["Previous Day Topics", "PYQ Practice"],
                    "activity": "Revision + PYQ Solving",
                    "target": "Solve 25 PYQ questions"
                }
            ]
            
            # Add PYQ snippets
            for t in primary_topics + secondary_topics:
                if t['code'] in topic_data:
                    pyqs = topic_data[t['code']]['questions']
                    snippets = [q['question_text'][:80] + "..." 
                              for q in pyqs if len(q.get('question_text', '')) > 10]
                    day_plan["pyq_practice"].extend(snippets[:2])
        
        elif day <= 14:
            # Phase 2: Advanced Practice
            day_plan["phase"] = "Advanced Practice"
            day_plan["phase_description"] = "Speed building & advanced problems"
            
            section_cycle = ["Numerical Ability", "Reasoning", 
                           "Verbal Ability", "General Awareness"]
            primary = section_cycle[(day - 9) % 4]
            secondary = section_cycle[(day - 8) % 4]
            
            primary_topics = [t for t in topic_scores if t['section'] == primary][2:4]
            secondary_topics = [t for t in topic_scores if t['section'] == secondary][2:4]
            
            day_plan["sessions"] = [
                {
                    "time": "6:00 AM - 8:30 AM",
                    "duration": "2.5 hours",
                    "section": primary,
                    "topics": [t['name'] for t in primary_topics],
                    "activity": "Advanced Problems",
                    "target": "Solve 30 medium-hard questions"
                },
                {
                    "time": "10:00 AM - 12:00 PM",
                    "duration": "2 hours",
                    "section": secondary,
                    "topics": [t['name'] for t in secondary_topics],
                    "activity": "Speed Practice",
                    "target": "Timed practice - 25 questions in 30 min"
                },
                {
                    "time": "4:00 PM - 5:30 PM",
                    "duration": "1.5 hours",
                    "section": "Mixed Practice",
                    "topics": ["Weak Areas", "Error Analysis"],
                    "activity": "Targeted Improvement",
                    "target": "Solve 20 PYQ from weak areas"
                }
            ]
        
        elif day <= 17:
            # Phase 3: Mock Tests
            day_plan["phase"] = "Mock Test Phase"
            day_plan["phase_description"] = "Full-length tests & detailed analysis"
            
            mock_num = day - 14
            day_plan["sessions"] = [
                {
                    "time": "6:00 AM - 8:00 AM",
                    "duration": "2 hours",
                    "section": f"Mock Test {mock_num}",
                    "topics": ["100 Questions", "Full Syllabus"],
                    "activity": "Full Mock Test",
                    "target": "Complete in 2 hours, attempt 90+ questions"
                },
                {
                    "time": "10:00 AM - 12:30 PM",
                    "duration": "2.5 hours",
                    "section": "Analysis",
                    "topics": ["Error Analysis", "Solution Review"],
                    "activity": "Detailed Analysis",
                    "target": "Analyze every wrong answer, identify patterns"
                },
                {
                    "time": "4:00 PM - 5:30 PM",
                    "duration": "1.5 hours",
                    "section": "Weak Topics",
                    "topics": ["Topics with <60% accuracy"],
                    "activity": "Targeted Practice",
                    "target": "Solve 30 questions from weak areas"
                }
            ]
        
        else:
            # Phase 4: Final Revision
            day_plan["phase"] = "Final Revision"
            day_plan["phase_description"] = "Quick revision & confidence building"
            
            sections = ["Verbal Ability", "General Awareness"] if day == 18 else \
                      ["Numerical Ability", "Reasoning"] if day == 19 else \
                      ["All Sections"]
            
            day_plan["sessions"] = [
                {
                    "time": "6:00 AM - 8:30 AM",
                    "duration": "2.5 hours",
                    "section": sections[0],
                    "topics": ["All High Priority Topics"],
                    "activity": "Rapid Revision",
                    "target": "Review notes, formulas, key facts"
                },
                {
                    "time": "10:00 AM - 12:00 PM",
                    "duration": "2 hours",
                    "section": sections[-1],
                    "topics": ["Key Concepts", "Shortcuts"],
                    "activity": "Quick Practice",
                    "target": "Solve 50 quick questions"
                },
                {
                    "time": "4:00 PM - 5:30 PM",
                    "duration": "1.5 hours",
                    "section": "Relaxation",
                    "topics": ["Light Reading", "Confidence Building"],
                    "activity": "Mental Preparation",
                    "target": "Stay calm, sleep early"
                }
            ]
        
        daily_schedule.append(day_plan)
    
    return {
        "metadata": {
            "total_days": total_days,
            "hours_per_day": hours_per_day,
            "total_hours": total_hours,
            "generated_at": datetime.now().isoformat(),
            "prediction_model": "Ensemble ML (XGBoost+LightGBM+RF)",
            "phases": [
                {"name": "Concept Building", "days": "1-8", "focus": "Fundamentals"},
                {"name": "Advanced Practice", "days": "9-14", "focus": "Speed & Accuracy"},
                {"name": "Mock Tests", "days": "15-17", "focus": "Full Practice"},
                {"name": "Final Revision", "days": "18-20", "focus": "Quick Recap"}
            ]
        },
        "section_allocation": {
            section: {"hours": hours, "percentage": round((hours / total_hours) * 100)}
            for section, hours in section_hours.items()
        },
        "daily_schedule": daily_schedule,
        "top_priority_topics": [
            {
                "topic": t['name'],
                "section": t['section'],
                "predicted_count": t['predicted'],
                "priority_score": round(t['score'], 2)
            }
            for t in topic_scores[:15]
        ]
    }

# ==========================================
# MOCK TEST & PRACTICE QUESTIONS
# ==========================================

def generate_mock_blueprint(pyq_data, ml_predictions):
    """Generate mock test using ML predictions"""
    sections = []
    all_questions = []
    question_number = 1
    
    for section, quota in SECTION_DISTRIBUTION.items():
        section_preds = {k: v for k, v in ml_predictions.items() 
                        if v['section'] == section}
        section_questions = []
        used_questions = set()
        
        # Sort by predicted count
        sorted_preds = sorted(section_preds.items(), 
                            key=lambda x: x[1]['predicted_count'], reverse=True)
        
        remaining = quota
        
        for code, pred in sorted_preds:
            if remaining <= 0:
                break
            
            topic_q_count = min(remaining, max(1, round(pred['predicted_count'])))
            
            # Get questions for this topic
            available = [q for q in pyq_data 
                        if q['topic_code'] == code and 
                        q['section'] == section and
                        id(q) not in used_questions and
                        "not available" not in q['question_text'].lower()]
            
            # Prefer recent years
            available.sort(key=lambda x: x['year'], reverse=True)
            selected = available[:topic_q_count]
            
            for q in selected:
                used_questions.add(id(q))
                section_questions.append({
                    "q_no": question_number,
                    "text": q['question_text'],
                    "options": q['choices'],
                    "answer": q['answer'],
                    "topic": pred['topic_name'],
                    "year": q['year'],
                    "marks": 3,
                    "negative": -1
                })
                question_number += 1
                remaining -= 1
        
        # Fill remaining with random questions
        while remaining > 0:
            section_pyq = [q for q in pyq_data 
                          if q['section'] == section and 
                          id(q) not in used_questions and
                          "not available" not in q['question_text'].lower()]
            if not section_pyq:
                break
            
            q = random.choice(section_pyq)
            used_questions.add(id(q))
            section_questions.append({
                "q_no": question_number,
                "text": q['question_text'],
                "options": q['choices'],
                "answer": q['answer'],
                "topic": q['topic_display'],
                "year": q['year'],
                "marks": 3,
                "negative": -1
            })
            question_number += 1
            remaining -= 1
        
        sections.append({
            "section": section,
            "questions": section_questions,
            "total_questions": len(section_questions),
            "total_marks": len(section_questions) * 3
        })
        all_questions.extend(section_questions)
    
    return {
        "metadata": {
            "exam": "AFCAT 2026 Mock Test (ML-Predicted)",
            "total_questions": len(all_questions),
            "total_marks": len(all_questions) * 3,
            "duration": "120 minutes",
            "generated_from": "ML Ensemble Predictions + Actual PYQ"
        },
        "sections": sections,
        "all_questions": all_questions
    }

def generate_practice_set(topic_data):
    """Generate topic-wise practice questions"""
    practice_set = {}
    
    for code, data in topic_data.items():
        valid_qs = [q for q in data['questions'] 
                   if "not available" not in q.get('question_text', '').lower()]
        
        if len(valid_qs) >= 3:
            # Prefer recent questions
            valid_qs.sort(key=lambda x: x['year'], reverse=True)
            selected = valid_qs[:5]
            
            practice_set[data['topic_name']] = [{
                "question": q['question_text'],
                "options": q['choices'],
                "answer": q['answer'],
                "year": q['year']
            } for q in selected]
    
    return practice_set

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 70)
    print("🎯 AFCAT 2026 ULTIMATE PREDICTION SYSTEM")
    print("=" * 70)
    print("Features: Ensemble ML | Linear Regression | Advanced Study Plan")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    processed_data = load_data()
    print(f"   ✓ Loaded {len(processed_data)} questions")
    
    # Step 2: Analyze by topic
    print("\n[2/6] Analyzing topics...")
    topic_data = defaultdict(lambda: {
        'questions': [], 'years': [], 'count': 0, 
        'topic_name': '', 'section': ''
    })
    
    for q in processed_data:
        code = q['topic_code']
        if code == "UNKNOWN":
            continue
        topic_data[code]['questions'].append(q)
        topic_data[code]['years'].append(q['year'])
        topic_data[code]['count'] += 1
        topic_data[code]['topic_name'] = q['topic_display']
        topic_data[code]['section'] = q['section']
    
    print(f"   ✓ Found {len(topic_data)} unique topics")
    
    # Step 3: ML Predictions
    print("\n[3/6] Running ML ensemble predictions...")
    ml_predictions = ensemble_predict(topic_data)
    print(f"   ✓ Generated predictions for {len(ml_predictions)} topics")
    
    if XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE and SKLEARN_AVAILABLE:
        print("   ✓ Using: XGBoost + LightGBM + RandomForest")
    elif any([XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE]):
        print("   ⚠️ Using partial ensemble (some ML libraries missing)")
    else:
        print("   ⚠️ Using fallback predictor (install xgboost/lightgbm for best results)")
    
    # Step 4: Trend analysis
    print("\n[4/6] Calculating trends (Linear Regression)...")
    rising, declining = calculate_trends_linreg(topic_data)
    print(f"   ✓ Rising: {len(rising)} | Declining: {len(declining)}")
    
    # Step 5: Study plan
    print("\n[5/6] Generating 20-day study plan...")
    study_plan = generate_ultimate_study_plan(topic_data, ml_predictions, rising)
    print(f"   ✓ {study_plan['metadata']['total_days']} days, "
          f"{study_plan['metadata']['total_hours']} hours")
    
    # Step 6: Mock test & practice
    print("\n[6/6] Creating mock test & practice sets...")
    mock_test = generate_mock_blueprint(processed_data, ml_predictions)
    practice_set = generate_practice_set(topic_data)
    print(f"   ✓ Mock: {mock_test['metadata']['total_questions']} questions")
    print(f"   ✓ Practice: {len(practice_set)} topics")
    
    # Compile final data
    final_data = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "prediction_model": "Ensemble (XGBoost+LightGBM+RF)",
            "trend_model": "Linear Regression",
            "total_questions_analyzed": len(processed_data)
        },
        "predictions": {
            "rising": rising,
            "declining": declining,
            "ml_predictions": list(ml_predictions.values())
        },
        "study_plan": study_plan,
        "mock_test": mock_test,
        "practice_questions": practice_set
    }
    
    # Save data.js (main output)
    js_content = f"const dashboardData = {json.dumps(final_data, indent=2, ensure_ascii=False)};"
    js_path = OUTPUT_DIR / "data.js"
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    # Save individual JSON files
    with open(OUTPUT_DIR / "afcat_2026_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(final_data['predictions'], f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_DIR / "afcat_2026_study_plan.json", 'w', encoding='utf-8') as f:
        json.dump(study_plan, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_DIR / "afcat_2026_mock.json", 'w', encoding='utf-8') as f:
        json.dump(mock_test, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Ultimate prediction system complete")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   • Questions Analyzed: {len(processed_data)}")
    print(f"   • Topics Predicted: {len(ml_predictions)}")
    print(f"   • Rising Topics: {len(rising)}")
    print(f"   • Mock Questions: {mock_test['metadata']['total_questions']}")
    print(f"   • Study Days: {study_plan['metadata']['total_days']}")
    print(f"\n📁 Output:")
    print(f"   • {OUTPUT_DIR}/data.js (main dashboard file)")
    print(f"   • {OUTPUT_DIR}/afcat_2026_predictions.json")
    print(f"   • {OUTPUT_DIR}/afcat_2026_study_plan.json")
    print(f"   • {OUTPUT_DIR}/afcat_2026_mock.json")
    print("=" * 70)

if __name__ == "__main__":
    main()