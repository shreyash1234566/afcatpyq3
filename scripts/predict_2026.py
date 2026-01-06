#!/usr/bin/env python3
"""
AFCAT 2026 PREDICTION SYSTEM - Complete Pipeline
Uses PYQ (Previous Year Questions) data to:
1. Train XGBoost model on historical data
2. Predict topics for 2026
3. Generate study plan
4. Create mock test blueprint
5. Generate sample questions

Author: AFCAT Prediction System
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ ML libraries not found. Using statistical prediction.")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SECTIONS = ["Verbal Ability", "General Awareness", "Numerical Ability", "Reasoning"]

SECTION_DISTRIBUTION_2026 = {
    "Verbal Ability": 30,
    "General Awareness": 25,
    "Numerical Ability": 20,
    "Reasoning": 25
}

TOPIC_TAXONOMY = {
    "Verbal Ability": [
        ("VA_COMP", "Reading Comprehension"),
        ("VA_CLOZE", "Cloze Test"),
        ("VA_ERR", "Error Detection"),
        ("VA_SENT", "Sentence Completion"),
        ("VA_REARR", "Sentence Rearrangement"),
        ("VA_SYN", "Synonyms"),
        ("VA_ANT", "Antonyms"),
        ("VA_OWS", "One Word Substitution"),
        ("VA_IDIOM", "Idioms & Phrases"),
        ("VA_GRAM", "Grammar"),
        ("VA_ANALOGY", "Verbal Analogy"),
    ],
    "General Awareness": [
        ("GA_HIST_ANC", "Ancient History"),
        ("GA_HIST_MED", "Medieval History"),
        ("GA_HIST_MOD", "Modern History"),
        ("GA_GEO_IND", "Indian Geography"),
        ("GA_GEO_WORLD", "World Geography"),
        ("GA_POLITY", "Polity & Governance"),
        ("GA_ECON", "Economy"),
        ("GA_ENV", "Environment"),
        ("GA_SCI", "Science & Technology"),
        ("GA_DEF", "Defence & Military"),
        ("GA_CULTURE", "Art & Culture"),
        ("GA_CURR", "Current Affairs"),
        ("GA_SPORTS", "Sports"),
        ("GA_AWARDS", "Awards & Honours"),
        ("GA_BOOKS", "Books & Authors"),
        ("GA_ORG", "Organizations"),
    ],
    "Numerical Ability": [
        ("NA_NUM", "Number System"),
        ("NA_DEC", "Decimal & Fraction"),
        ("NA_SIM", "Simplification"),
        ("NA_RAT", "Ratio & Proportion"),
        ("NA_AVG", "Average"),
        ("NA_PER", "Percentage"),
        ("NA_PL", "Profit & Loss"),
        ("NA_SI", "Simple Interest"),
        ("NA_CI", "Compound Interest"),
        ("NA_TW", "Time & Work"),
        ("NA_SPD", "Speed, Distance, Time"),
        ("NA_MIX", "Mixture & Alligation"),
        ("NA_AREA", "Area & Perimeter"),
        ("NA_MENSA", "Mensuration"),
        ("NA_STAT", "Statistics"),
        ("NA_HCF", "HCF & LCM"),
        ("NA_ALG", "Algebra"),
    ],
    "Reasoning": [
        ("RM_VR_CLASS", "Classification"),
        ("RM_VR_ANALOGY", "Verbal Analogy"),
        ("RM_VR_SERIES", "Series"),
        ("RM_VR_CODING", "Coding-Decoding"),
        ("RM_NV_VENN", "Venn Diagrams"),
        ("RM_VR_SYLL", "Syllogism"),
        ("RM_VR_LOG", "Logical Reasoning"),
        ("RM_NV_SPATIAL", "Spatial Ability"),
        ("RM_NV_PATTERN", "Pattern/Figure Series"),
        ("RM_NV_MIRROR", "Mirror & Water Images"),
        ("RM_NV_CUBES", "Cubes & Dice"),
        ("RM_NV_ORIENT", "Direction Sense"),
        ("RM_VR_BLOOD", "Blood Relations"),
        ("RM_VR_ORDER", "Order & Ranking"),
    ]
}

# Study hours per topic (based on difficulty)
HOURS_PER_TOPIC = {
    "Easy": 3,
    "Medium": 5,
    "Hard": 8
}

# ============================================================================
# DATA LOADER
# ============================================================================

class PYQDataLoader:
    """Load and process PYQ data from Q.json."""
    
    def __init__(self, json_path: str = "data/processed/Q.json"):
        self.json_path = json_path
        self.questions = []
        self.by_year = defaultdict(list)
        self.by_section = defaultdict(list)
        self.by_topic = defaultdict(list)
        
    def load(self) -> int:
        """Load questions from JSON."""
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.questions = data if isinstance(data, list) else data.get('questions', [])
        
        # Index by year, section, topic
        for q in self.questions:
            # Extract year from file_name
            file_name = q.get('file_name', '')
            year = self._extract_year(file_name)
            
            section = q.get('section', 'Unknown')
            topic = q.get('topic', 'Unknown')
            topic_code = q.get('topic_code', 'Unknown')
            
            q['year'] = year
            
            self.by_year[year].append(q)
            self.by_section[section].append(q)
            self.by_topic[topic_code].append(q)
        
        return len(self.questions)
    
    def _extract_year(self, file_name: str) -> int:
        """Extract year from file name like AFCAT_2011_Official_Paper2.pdf."""
        import re
        match = re.search(r'20\d{2}', file_name)
        if match:
            return int(match.group())
        return 2020  # Default
    
    def get_year_range(self) -> Tuple[int, int]:
        """Get min and max years in data."""
        years = [y for y in self.by_year.keys() if isinstance(y, int)]
        return min(years) if years else 2011, max(years) if years else 2025
    
    def get_topic_frequency_by_year(self) -> Dict[str, Dict[int, int]]:
        """Get topic frequency for each year."""
        result = defaultdict(lambda: defaultdict(int))
        
        for q in self.questions:
            topic_code = q.get('topic_code', 'Unknown')
            year = q.get('year', 2020)
            result[topic_code][year] += 1
        
        return dict(result)
    
    def get_section_distribution(self) -> Dict[str, int]:
        """Get section distribution."""
        return {section: len(questions) for section, questions in self.by_section.items()}
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get difficulty distribution."""
        difficulties = Counter(q.get('difficulty', 'Medium') for q in self.questions)
        return dict(difficulties)


# ============================================================================
# XGBOOST TOPIC PREDICTOR
# ============================================================================

class XGBoostTopicPredictor:
    """Predict 2026 topic frequencies using XGBoost."""
    
    def __init__(self):
        self.model = None
        self.topic_history = {}
        self.predictions = {}
        
    def prepare_features(self, topic_freq: Dict[int, int], years: List[int]) -> np.ndarray:
        """Create feature vector for a topic."""
        
        frequencies = [topic_freq.get(y, 0) for y in years]
        
        # Features:
        # 1. Average frequency
        avg_freq = np.mean(frequencies) if frequencies else 0
        
        # 2. Recent frequency (last 2 years)
        recent_freq = np.mean(frequencies[-2:]) if len(frequencies) >= 2 else avg_freq
        
        # 3. Trend (linear regression slope)
        if len(frequencies) >= 3:
            x = np.arange(len(frequencies))
            trend = np.polyfit(x, frequencies, 1)[0]
        else:
            trend = 0
        
        # 4. Volatility (std dev)
        volatility = np.std(frequencies) if len(frequencies) >= 2 else 0
        
        # 5. Max frequency
        max_freq = max(frequencies) if frequencies else 0
        
        # 6. Min frequency
        min_freq = min(frequencies) if frequencies else 0
        
        # 7. Consecutive appearances
        consecutive = sum(1 for f in frequencies if f > 0)
        
        # 8. Recent growth rate
        if len(frequencies) >= 2 and frequencies[-2] > 0:
            growth_rate = (frequencies[-1] - frequencies[-2]) / frequencies[-2]
        else:
            growth_rate = 0
        
        return np.array([avg_freq, recent_freq, trend, volatility, max_freq, 
                         min_freq, consecutive, growth_rate])
    
    def train(self, topic_frequencies: Dict[str, Dict[int, int]], years: List[int]):
        """Train XGBoost model on historical data."""
        
        self.topic_history = topic_frequencies
        
        if not HAS_ML:
            logger.info("Using statistical prediction (no ML libraries)")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for topic, freq_by_year in topic_frequencies.items():
            # Use years[:-1] for features, years[-1] for target
            if len(years) >= 2:
                features = self.prepare_features(freq_by_year, years[:-1])
                target = freq_by_year.get(years[-1], 0)
                X.append(features)
                y.append(target)
        
        if len(X) < 5:
            logger.warning("Not enough data for ML training, using statistical prediction")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X, y)
        logger.info(f"✓ XGBoost model trained on {len(X)} topics")
    
    def predict_2026(self, years: List[int]) -> Dict[str, float]:
        """Predict topic frequencies for 2026."""
        
        predictions = {}
        
        for topic, freq_by_year in self.topic_history.items():
            if self.model and HAS_ML:
                # ML prediction
                features = self.prepare_features(freq_by_year, years)
                pred = self.model.predict([features])[0]
                predictions[topic] = max(0, pred)
            else:
                # Statistical prediction (weighted average with trend)
                frequencies = [freq_by_year.get(y, 0) for y in years]
                if frequencies:
                    # Weight recent years more heavily
                    weights = [1, 1, 2, 2, 3, 3][:len(frequencies)]
                    weights = weights[-len(frequencies):]
                    weighted_avg = np.average(frequencies, weights=weights)
                    
                    # Apply trend
                    if len(frequencies) >= 3:
                        x = np.arange(len(frequencies))
                        trend = np.polyfit(x, frequencies, 1)[0]
                        pred = weighted_avg + trend
                    else:
                        pred = weighted_avg
                    
                    predictions[topic] = max(0, pred)
                else:
                    predictions[topic] = 0
        
        self.predictions = predictions
        return predictions


# ============================================================================
# 2026 PREDICTION GENERATOR
# ============================================================================

class AFCAT2026Predictor:
    """Generate complete 2026 predictions."""
    
    def __init__(self, data_loader: PYQDataLoader):
        self.loader = data_loader
        self.topic_predictor = XGBoostTopicPredictor()
        self.predictions = {}
        
    def run_full_prediction(self) -> Dict:
        """Run complete prediction pipeline."""
        
        print("\n" + "=" * 80)
        print("🚀 AFCAT 2026 PREDICTION PIPELINE")
        print("=" * 80)
        
        # Step 1: Analyze historical data
        print("\n📊 Step 1: Analyzing Historical PYQ Data...")
        topic_freq = self.loader.get_topic_frequency_by_year()
        min_year, max_year = self.loader.get_year_range()
        years = list(range(min_year, max_year + 1))
        
        print(f"   ✓ Data range: {min_year} - {max_year}")
        print(f"   ✓ Total questions: {len(self.loader.questions)}")
        print(f"   ✓ Topics found: {len(topic_freq)}")
        
        # Step 2: Train XGBoost model
        print("\n🤖 Step 2: Training XGBoost Model...")
        self.topic_predictor.train(topic_freq, years)
        
        # Step 3: Predict 2026 topics
        print("\n🔮 Step 3: Predicting 2026 Topic Distribution...")
        raw_predictions = self.topic_predictor.predict_2026(years)
        
        # Normalize to section quotas
        section_predictions = self._normalize_to_sections(raw_predictions)
        
        # Step 4: Build complete prediction
        print("\n📈 Step 4: Building Complete 2026 Blueprint...")
        self.predictions = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_source": "PYQ 2011-2025",
                "total_questions_analyzed": len(self.loader.questions),
                "years_covered": f"{min_year}-{max_year}",
                "target_year": 2026,
                "confidence": 0.75
            },
            "exam_structure": {
                "total_questions": 100,
                "total_marks": 300,
                "correct_marks": 3,
                "negative_marks": -1,
                "duration_minutes": 120
            },
            "section_distribution": SECTION_DISTRIBUTION_2026,
            "topic_predictions": section_predictions,
            "difficulty_prediction": self._predict_difficulty(),
            "rising_topics": self._get_rising_topics(topic_freq, years),
            "declining_topics": self._get_declining_topics(topic_freq, years),
            "high_priority_topics": self._get_high_priority(section_predictions)
        }
        
        return self.predictions
    
    def _normalize_to_sections(self, raw_predictions: Dict[str, float]) -> Dict[str, List[Dict]]:
        """Normalize predictions to section quotas."""
        
        result = {}
        
        for section, quota in SECTION_DISTRIBUTION_2026.items():
            section_topics = []
            
            # Get topics for this section from taxonomy
            if section in TOPIC_TAXONOMY:
                for topic_code, topic_name in TOPIC_TAXONOMY[section]:
                    pred_count = raw_predictions.get(topic_code, 0)
                    
                    # Also check variations
                    if pred_count == 0:
                        for key in raw_predictions:
                            if topic_code in key or key in topic_code:
                                pred_count = max(pred_count, raw_predictions[key])
                    
                    section_topics.append({
                        "topic_code": topic_code,
                        "topic_name": topic_name,
                        "predicted_count": round(pred_count, 1),
                        "confidence": 0.7 + (0.2 if pred_count > 2 else 0)
                    })
            
            # Normalize to quota
            total_pred = sum(t['predicted_count'] for t in section_topics)
            if total_pred > 0:
                scale = quota / total_pred
                for t in section_topics:
                    t['predicted_count'] = round(t['predicted_count'] * scale, 1)
            
            # Sort by predicted count
            section_topics.sort(key=lambda x: -x['predicted_count'])
            result[section] = section_topics
        
        return result
    
    def _predict_difficulty(self) -> Dict[str, float]:
        """Predict difficulty distribution."""
        
        hist_diff = self.loader.get_difficulty_distribution()
        total = sum(hist_diff.values())
        
        if total > 0:
            return {
                "Easy": round(hist_diff.get('Easy', 0) / total * 100, 1),
                "Medium": round(hist_diff.get('Medium', 0) / total * 100, 1),
                "Hard": round(hist_diff.get('Hard', 0) / total * 100, 1)
            }
        
        return {"Easy": 30, "Medium": 50, "Hard": 20}
    
    def _get_rising_topics(self, topic_freq: Dict, years: List[int]) -> List[Dict]:
        """Get topics with rising trend."""
        
        rising = []
        
        for topic, freq_by_year in topic_freq.items():
            frequencies = [freq_by_year.get(y, 0) for y in years]
            
            if len(frequencies) >= 3:
                # Check if last 2 years > average of first 2 years
                recent = np.mean(frequencies[-2:])
                earlier = np.mean(frequencies[:2])
                
                if earlier > 0 and recent > earlier * 1.3:  # 30% increase
                    growth_pct = min(int((recent/earlier - 1) * 100), 500)  # Cap at 500%
                    rising.append({
                        "topic_code": topic,
                        "trend": "rising",
                        "growth": f"+{growth_pct}%"
                    })
        
        return rising[:10]
    
    def _get_declining_topics(self, topic_freq: Dict, years: List[int]) -> List[Dict]:
        """Get topics with declining trend."""
        
        declining = []
        
        for topic, freq_by_year in topic_freq.items():
            frequencies = [freq_by_year.get(y, 0) for y in years]
            
            if len(frequencies) >= 3:
                recent = np.mean(frequencies[-2:])
                earlier = np.mean(frequencies[:2])
                
                if earlier > 0 and recent < earlier * 0.7:  # 30% decrease
                    declining.append({
                        "topic_code": topic,
                        "trend": "declining",
                        "decline": f"-{int((1 - recent/earlier) * 100)}%"
                    })
        
        return declining[:10]
    
    def _get_high_priority(self, section_predictions: Dict) -> List[Dict]:
        """Get high priority topics for study."""
        
        priority = []
        
        for section, topics in section_predictions.items():
            for t in topics[:5]:  # Top 5 per section
                if t['predicted_count'] >= 2:
                    priority.append({
                        "section": section,
                        "topic_code": t['topic_code'],
                        "topic_name": t['topic_name'],
                        "expected_questions": t['predicted_count'],
                        "priority": "HIGH" if t['predicted_count'] >= 4 else "MEDIUM"
                    })
        
        priority.sort(key=lambda x: -x['expected_questions'])
        return priority[:20]


# ============================================================================
# STUDY PLAN GENERATOR
# ============================================================================

class StudyPlanGenerator:
    """Generate optimized study plan for 2026."""
    
    def __init__(self, predictions: Dict):
        self.predictions = predictions
        
    def generate(self, total_days: int = 60) -> Dict:
        """Generate study plan."""
        
        print("\n📚 Generating Optimized Study Plan...")
        
        topics = self.predictions.get('topic_predictions', {})
        priority_topics = self.predictions.get('high_priority_topics', [])
        
        # Calculate hours per section
        section_hours = {}
        topic_hours = []
        
        for section, section_topics in topics.items():
            section_total = 0
            
            for t in section_topics:
                count = t['predicted_count']
                
                # Hours based on importance
                if count >= 4:
                    hours = 8
                    priority = "HIGH"
                elif count >= 2:
                    hours = 5
                    priority = "MEDIUM"
                else:
                    hours = 2
                    priority = "LOW"
                
                topic_hours.append({
                    "section": section,
                    "topic_code": t['topic_code'],
                    "topic_name": t['topic_name'],
                    "expected_questions": count,
                    "study_hours": hours,
                    "priority": priority
                })
                
                section_total += hours
            
            section_hours[section] = section_total
        
        # Sort by priority
        topic_hours.sort(key=lambda x: (-x['expected_questions'], x['section']))
        
        # Daily schedule
        total_hours = sum(section_hours.values())
        hours_per_day = total_hours / total_days
        
        study_plan = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_study_days": total_days,
                "total_study_hours": total_hours,
                "hours_per_day": round(hours_per_day, 1)
            },
            "section_allocation": {
                section: {
                    "hours": hours,
                    "percentage": round(hours / total_hours * 100, 1),
                    "days": round(hours / hours_per_day, 1)
                }
                for section, hours in section_hours.items()
            },
            "topic_wise_plan": topic_hours[:30],  # Top 30 topics
            "weekly_schedule": self._create_weekly_schedule(topic_hours, hours_per_day),
            "revision_days": 7,  # Last 7 days for revision
            "mock_test_days": [total_days - 14, total_days - 7, total_days - 1]
        }
        
        return study_plan
    
    def _create_weekly_schedule(self, topic_hours: List[Dict], hours_per_day: float) -> List[Dict]:
        """Create weekly schedule."""
        
        weeks = []
        current_week = []
        week_hours = 0
        week_num = 1
        
        for topic in topic_hours:
            current_week.append(topic['topic_code'])
            week_hours += topic['study_hours']
            
            if week_hours >= hours_per_day * 7:  # One week worth
                weeks.append({
                    "week": week_num,
                    "topics": current_week,
                    "hours": round(week_hours, 1)
                })
                current_week = []
                week_hours = 0
                week_num += 1
        
        if current_week:
            weeks.append({
                "week": week_num,
                "topics": current_week,
                "hours": round(week_hours, 1)
            })
        
        return weeks


# ============================================================================
# MOCK TEST BLUEPRINT GENERATOR
# ============================================================================

class MockTestGenerator:
    """Generate mock test blueprint for 2026."""
    
    def __init__(self, predictions: Dict, questions: List[Dict]):
        self.predictions = predictions
        self.questions = questions
        
    def generate(self) -> Dict:
        """Generate mock test blueprint."""
        
        print("\n📝 Generating Mock Test Blueprint...")
        
        blueprint = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_questions": 100,
                "duration_minutes": 120,
                "total_marks": 300
            },
            "sections": []
        }
        
        topics = self.predictions.get('topic_predictions', {})
        
        for section, quota in SECTION_DISTRIBUTION_2026.items():
            section_blueprint = {
                "section": section,
                "questions": quota,
                "topics": []
            }
            
            if section in topics:
                remaining = quota
                for t in topics[section]:
                    count = min(int(round(t['predicted_count'])), remaining)
                    if count > 0:
                        section_blueprint['topics'].append({
                            "topic_code": t['topic_code'],
                            "topic_name": t['topic_name'],
                            "questions": count,
                            "difficulty_mix": {
                                "Easy": int(count * 0.3),
                                "Medium": int(count * 0.5),
                                "Hard": count - int(count * 0.3) - int(count * 0.5)
                            }
                        })
                        remaining -= count
                    
                    if remaining <= 0:
                        break
            
            blueprint['sections'].append(section_blueprint)
        
        return blueprint
    
    def generate_sample_questions(self, count: int = 20) -> List[Dict]:
        """Generate sample questions from PYQ bank."""
        
        print(f"\n🎯 Selecting {count} Sample Questions...")
        
        samples = []
        high_priority = self.predictions.get('high_priority_topics', [])
        
        priority_codes = [t['topic_code'] for t in high_priority[:10]]
        
        # Select from high priority topics
        for q in self.questions:
            if len(samples) >= count:
                break
            
            topic_code = q.get('topic_code', '')
            
            # Check if matches priority topics
            matches = any(p in topic_code or topic_code in p for p in priority_codes)
            
            if matches:
                samples.append({
                    "question_number": len(samples) + 1,
                    "section": q.get('section'),
                    "topic": q.get('topic'),
                    "question_text": q.get('question_text'),
                    "choices": q.get('choices', []),
                    "difficulty": q.get('difficulty', 'Medium'),
                    "source": "PYQ Bank"
                })
        
        # Fill remaining with random selection
        if len(samples) < count:
            import random
            remaining = count - len(samples)
            random_qs = random.sample(self.questions, min(remaining, len(self.questions)))
            
            for q in random_qs:
                samples.append({
                    "question_number": len(samples) + 1,
                    "section": q.get('section'),
                    "topic": q.get('topic'),
                    "question_text": q.get('question_text'),
                    "choices": q.get('choices', []),
                    "difficulty": q.get('difficulty', 'Medium'),
                    "source": "PYQ Bank"
                })
        
        return samples[:count]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """Run complete 2026 prediction pipeline."""
    
    print("\n" + "=" * 80)
    print("🎯 AFCAT 2026 COMPLETE PREDICTION SYSTEM")
    print("=" * 80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load PYQ data
    print("\n" + "-" * 80)
    print("📂 STEP 1: Loading PYQ Data...")
    
    loader = PYQDataLoader("data/processed/Q.json")
    total = loader.load()
    
    print(f"   ✓ Loaded {total} questions")
    print(f"   ✓ Years: {loader.get_year_range()}")
    print(f"   ✓ Sections: {list(loader.by_section.keys())}")
    
    # Step 2: Generate predictions
    print("\n" + "-" * 80)
    print("🔮 STEP 2: Generating 2026 Predictions...")
    
    predictor = AFCAT2026Predictor(loader)
    predictions = predictor.run_full_prediction()
    
    # Step 3: Generate study plan
    print("\n" + "-" * 80)
    print("📚 STEP 3: Generating Study Plan...")
    
    study_gen = StudyPlanGenerator(predictions)
    study_plan = study_gen.generate(total_days=60)
    
    # Step 4: Generate mock blueprint
    print("\n" + "-" * 80)
    print("📝 STEP 4: Generating Mock Test Blueprint...")
    
    mock_gen = MockTestGenerator(predictions, loader.questions)
    mock_blueprint = mock_gen.generate()
    sample_questions = mock_gen.generate_sample_questions(30)
    
    # Step 5: Save all outputs
    print("\n" + "-" * 80)
    print("💾 STEP 5: Saving Outputs...")
    
    output_dir = Path("output/predictions_2026")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save predictions
    with open(output_dir / "afcat_2026_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(predictions), f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {output_dir}/afcat_2026_predictions.json")
    
    # Save study plan
    with open(output_dir / "afcat_2026_study_plan.json", 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(study_plan), f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {output_dir}/afcat_2026_study_plan.json")
    
    # Save mock blueprint
    with open(output_dir / "afcat_2026_mock_blueprint.json", 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(mock_blueprint), f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {output_dir}/afcat_2026_mock_blueprint.json")
    
    # Save sample questions
    with open(output_dir / "afcat_2026_sample_questions.json", 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(sample_questions), f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {output_dir}/afcat_2026_sample_questions.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 2026 PREDICTION SUMMARY")
    print("=" * 80)
    
    print("\n📁 Section Distribution:")
    for section, quota in SECTION_DISTRIBUTION_2026.items():
        print(f"   {section:<25}: {quota:>3} questions")
    
    print("\n🎯 Top 10 High Priority Topics:")
    for i, topic in enumerate(predictions.get('high_priority_topics', [])[:10], 1):
        print(f"   {i:>2}. {topic['topic_name']:<30} ({topic['expected_questions']:.1f} Q)")
    
    print("\n📈 Rising Topics (Focus More):")
    for topic in predictions.get('rising_topics', [])[:5]:
        print(f"   📈 {topic['topic_code']} {topic['growth']}")
    
    print("\n📉 Declining Topics (Lower Priority):")
    for topic in predictions.get('declining_topics', [])[:5]:
        print(f"   📉 {topic['topic_code']} {topic['decline']}")
    
    print("\n📚 Study Plan Summary:")
    print(f"   Total Days: {study_plan['metadata']['total_study_days']}")
    print(f"   Total Hours: {study_plan['metadata']['total_study_hours']}")
    print(f"   Hours/Day: {study_plan['metadata']['hours_per_day']}")
    
    for section, alloc in study_plan['section_allocation'].items():
        print(f"   {section:<25}: {alloc['hours']:>3}h ({alloc['percentage']}%)")
    
    print("\n" + "=" * 80)
    print("✅ PREDICTION PIPELINE COMPLETE!")
    print(f"📂 All files saved to: {output_dir}")
    print("=" * 80)
    
    return predictions, study_plan, mock_blueprint, sample_questions


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    run_complete_pipeline()
