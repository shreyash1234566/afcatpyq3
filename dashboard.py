"""
Prediction Dashboard
====================
Generates comprehensive predictions and study plans for AFCAT 2026.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EXAM_CONFIG, HISTORICAL_FREQUENCIES, TOPIC_TAXONOMY,
    OUTPUT_DIR, PREDICTIONS_DIR, REPORTS_DIR
)
from utils.data_structures import (
    Section, TopicFrequency, TopicPrediction, SectionPrediction,
    ExamPrediction, MockTestBlueprint, StudyPlan, TrendDirection
)
from analysis.topic_analyzer import TopicAnalyzer
from analysis.trend_detector import TrendDetector, get_hot_topics, get_cold_topics
from models.topic_predictor import TopicPredictor
from models.difficulty_model import DifficultyClassifier, estimate_paper_difficulty

logger = logging.getLogger(__name__)


class PredictionDashboard:
    """
    Central dashboard for generating AFCAT 2026 predictions.
    """
    
    def __init__(self, target_year: int = 2026):
        self.target_year = target_year
        self.analyzer = TopicAnalyzer()
        self.trend_detector = TrendDetector()
        self.topic_predictor = TopicPredictor()
        self.difficulty_classifier = DifficultyClassifier()
        
        # Load historical data
        self._load_historical_data()
        
    def _load_historical_data(self):
        """Load historical frequency data from config."""
        years = list(range(2020, 2026))
        self.analyzer.load_from_config(HISTORICAL_FREQUENCIES, years)
        
    def generate_full_prediction(self) -> ExamPrediction:
        """
        Generate complete prediction for AFCAT 2026.
        """
        logger.info(f"Generating predictions for AFCAT {self.target_year}")
        
        # Train topic predictor
        self.topic_predictor.fit(self.analyzer.topic_frequencies)
        
        # Get predictions
        topic_predictions = self.topic_predictor.predict(
            self.analyzer.topic_frequencies,
            self.target_year
        )
        
        # Organize by section
        section_predictions = {}
        
        for section in Section:
            section_topics = [
                tp for tp in topic_predictions
                if tp.section == section
            ]
            
            # Get section config
            section_key = section.value
            section_config = EXAM_CONFIG['sections'].get(section_key, {})
            total_questions = section_config.get('questions', 25)
            
            # Predict section difficulty
            topic_dist = {tp.topic: int(tp.predicted_count) for tp in section_topics}
            difficulty_result = self.difficulty_classifier.predict_section_difficulty(
                section, topic_dist
            )
            
            section_predictions[section] = SectionPrediction(
                section=section,
                total_questions=total_questions,
                predicted_difficulty=difficulty_result['difficulty'],
                topic_predictions=section_topics,
                good_attempts_expected=difficulty_result['good_attempts_expected']
            )
            
        # Calculate overall confidence
        all_confidences = [tp.confidence for tp in topic_predictions]
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
        
        # Generate study plan
        study_plan = self._generate_study_plan(topic_predictions)
        
        # Create prediction object
        prediction = ExamPrediction(
            target_year=self.target_year,
            generated_at=datetime.now(),
            section_predictions=section_predictions,
            overall_confidence=overall_confidence,
            predicted_cutoff_range=(155, 165),
            study_plan=study_plan
        )
        
        return prediction
    
    def _generate_study_plan(
        self,
        predictions: List[TopicPrediction],
        total_hours: int = 200
    ) -> Dict:
        """Generate study plan based on predictions."""
        
        # Get priorities
        priorities = self.analyzer.get_study_priorities(total_hours)
        
        # High-yield clusters
        high_yield = [
            p['topic'] for p in priorities
            if p['roi_score'] > 2.0
        ][:10]
        
        # Section-wise allocation
        section_hours = {
            'numerical_ability': total_hours * 0.30,
            'verbal_ability': total_hours * 0.25,
            'reasoning': total_hours * 0.25,
            'general_awareness': total_hours * 0.20
        }
        
        # Daily schedule (assuming 3 hours/day for 60 days)
        daily_schedule = {
            'weekday': [
                {'time': '6:00-7:30', 'activity': 'Numerical Ability Practice'},
                {'time': '18:00-19:00', 'activity': 'General Awareness Revision'},
                {'time': '21:00-22:00', 'activity': 'Mock Test / Analysis'}
            ],
            'weekend': [
                {'time': '6:00-8:00', 'activity': 'Full Mock Test'},
                {'time': '10:00-12:00', 'activity': 'Weak Area Focus'},
                {'time': '16:00-18:00', 'activity': 'Current Affairs Deep Dive'}
            ]
        }
        
        # Weekly goals
        weekly_goals = [
            "Complete 3 full-length mock tests",
            "Master 2 new math topics",
            "Learn 50 new vocabulary words (OLQ focus)",
            "Cover all current affairs from last 2 weeks",
            "Practice 100 reasoning questions"
        ]
        
        # Revision schedule
        revision_schedule = {
            'week_1': ['Speed/Distance', 'Time/Work', 'Defense GK'],
            'week_2': ['Profit/Loss', 'Percentage', 'Sports GK'],
            'week_3': ['Cloze Test', 'Vocabulary', 'History'],
            'week_4': ['Spatial Reasoning', 'Venn Diagrams', 'Geography'],
            'final_week': ['All Mock Tests', 'Weak Areas', 'Current Affairs']
        }
        
        return {
            'total_hours': total_hours,
            'section_allocation': section_hours,
            'high_yield_clusters': high_yield,
            'top_priority_topics': [p for p in priorities[:15]],
            'daily_schedule': daily_schedule,
            'weekly_goals': weekly_goals,
            'revision_schedule': revision_schedule
        }
    
    def generate_mock_blueprint(self) -> MockTestBlueprint:
        """Generate blueprint for mock test based on predictions."""
        prediction = self.generate_full_prediction()
        
        sections = {}
        
        for section, sec_pred in prediction.section_predictions.items():
            topic_counts = {}
            total_section_qs = 0
            
            for tp in sec_pred.topic_predictions:
                count = round(tp.predicted_count)
                if count > 0:
                    topic_counts[tp.topic] = count
                    total_section_qs += count
                    
            # Adjust to match expected total
            expected = sec_pred.total_questions
            if total_section_qs != expected and topic_counts:
                # Scale proportionally
                scale = expected / total_section_qs
                for topic in topic_counts:
                    topic_counts[topic] = max(1, round(topic_counts[topic] * scale))
                    
            sections[section] = topic_counts
            
        return MockTestBlueprint(
            name=f"AFCAT {self.target_year} Predicted Blueprint",
            based_on_year=self.target_year,
            sections=sections,
            total_questions=100
        )
    
    def get_trend_insights(self) -> Dict:
        """Get trend analysis insights."""
        trend_analysis = self.trend_detector.analyze_all_trends(
            self.analyzer.topic_frequencies
        )
        
        hot_topics = get_hot_topics(trend_analysis, top_n=10)
        cold_topics = get_cold_topics(trend_analysis, top_n=5)
        
        return {
            'hot_topics': hot_topics,
            'cold_topics': cold_topics,
            'full_analysis': trend_analysis
        }
    
    def export_predictions(self, output_dir: Optional[Path] = None):
        """Export all predictions to files."""
        output_dir = output_dir or PREDICTIONS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate prediction
        prediction = self.generate_full_prediction()
        
        # Save JSON
        json_path = output_dir / f"afcat_{self.target_year}_prediction.json"
        prediction.save_json(str(json_path))
        
        # Save blueprint
        blueprint = self.generate_mock_blueprint()
        blueprint_path = output_dir / f"afcat_{self.target_year}_blueprint.json"
        with open(blueprint_path, 'w') as f:
            json.dump({
                'name': blueprint.name,
                'based_on_year': blueprint.based_on_year,
                'total_questions': blueprint.total_questions,
                'sections': {
                    s.value: topics for s, topics in blueprint.sections.items()
                }
            }, f, indent=2)
            
        # Save study plan
        plan_path = output_dir / f"afcat_{self.target_year}_study_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(prediction.study_plan, f, indent=2)
            
        logger.info(f"Predictions exported to {output_dir}")
        
        return {
            'prediction': str(json_path),
            'blueprint': str(blueprint_path),
            'study_plan': str(plan_path)
        }


def print_prediction_report(prediction: ExamPrediction):
    """Print formatted prediction report."""
    print("\n" + "=" * 80)
    print(f"{'AFCAT ' + str(prediction.target_year) + ' PREDICTION REPORT':^80}")
    print("=" * 80)
    print(f"\nGenerated: {prediction.generated_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"Overall Confidence: {prediction.overall_confidence:.1%}")
    print(f"Predicted Cutoff Range: {prediction.predicted_cutoff_range[0]} - {prediction.predicted_cutoff_range[1]}")
    
    for section, sec_pred in prediction.section_predictions.items():
        print(f"\n{'─' * 80}")
        print(f"📚 {section.value.upper().replace('_', ' ')}")
        print(f"{'─' * 80}")
        print(f"   Questions: {sec_pred.total_questions}")
        print(f"   Difficulty: {sec_pred.predicted_difficulty.upper()}")
        print(f"   Good Attempts Expected: {sec_pred.good_attempts_expected}")
        
        print("\n   Top Topics to Focus:")
        for i, tp in enumerate(sec_pred.topic_predictions[:5], 1):
            trend_icon = "📈" if tp.trend == TrendDirection.RISING else ("📉" if tp.trend == TrendDirection.DECLINING else "➡️")
            print(f"   {i}. {tp.topic.replace('_', ' ').title()}: "
                  f"{tp.predicted_count:.1f} Qs ({tp.confidence:.0%} conf.) {trend_icon}")
                  
    print("\n" + "=" * 80)
    print("HIGH-YIELD STUDY CLUSTERS")
    print("=" * 80)
    
    study_plan = prediction.study_plan
    if study_plan.get('high_yield_clusters'):
        for i, topic in enumerate(study_plan['high_yield_clusters'][:10], 1):
            print(f"   {i}. {topic.replace('_', ' ').title()}")
            
    print("\n" + "=" * 80)


def generate_html_report(prediction: ExamPrediction, output_path: Path, question_bank_data: Optional[Dict] = None):
    """
    Generate HTML report for predictions with collapsible question bank dropdowns.
    
    Args:
        prediction: The ExamPrediction object
        output_path: Path to save the HTML file
        question_bank_data: Optional dict of topic -> list of previous questions
    """
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AFCAT {year} Prediction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f0f4f8; }}
        h1 {{ color: #1a365d; text-align: center; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #2d3748; border-bottom: 2px solid #4299e1; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #4299e1; color: white; }}
        tr:hover {{ background: #edf2f7; }}
        .confidence-high {{ color: #38a169; font-weight: bold; }}
        .confidence-medium {{ color: #d69e2e; }}
        .confidence-low {{ color: #e53e3e; }}
        .trend-rising {{ color: #38a169; }}
        .trend-declining {{ color: #e53e3e; }}
        .trend-stable {{ color: #718096; }}
        .summary-box {{ background: #ebf8ff; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        
        /* Dropdown styles */
        .topic-dropdown {{ 
            margin: 10px 0; 
            border: 1px solid #e2e8f0; 
            border-radius: 8px;
            overflow: hidden;
        }}
        .topic-dropdown summary {{ 
            padding: 12px 15px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer; 
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
            list-style: none;
        }}
        .topic-dropdown summary::-webkit-details-marker {{ display: none; }}
        .topic-dropdown summary::after {{ 
            content: '▼'; 
            font-size: 12px;
            transition: transform 0.3s;
        }}
        .topic-dropdown[open] summary::after {{ transform: rotate(180deg); }}
        .topic-dropdown[open] summary {{ background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); }}
        
        .topic-meta {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            font-weight: normal;
        }}
        .topic-meta span {{ 
            background: rgba(255,255,255,0.2); 
            padding: 2px 8px; 
            border-radius: 4px; 
        }}
        
        .questions-list {{ 
            padding: 15px; 
            background: #f7fafc;
            max-height: 400px;
            overflow-y: auto;
        }}
        .prev-question {{ 
            background: white; 
            padding: 12px 15px; 
            margin: 8px 0; 
            border-radius: 6px;
            border-left: 4px solid #4299e1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .prev-question .q-text {{ margin-bottom: 8px; font-weight: 500; }}
        .prev-question .options {{ 
            margin-left: 15px; 
            font-size: 0.95em; 
            color: #4a5568;
        }}
        .prev-question .q-meta {{ 
            font-size: 0.85em; 
            color: #718096; 
            margin-top: 8px;
            display: flex;
            gap: 15px;
        }}
        .prev-question .q-meta .year {{ color: #4299e1; font-weight: 500; }}
        .prev-question .q-meta .difficulty-easy {{ color: #38a169; }}
        .prev-question .q-meta .difficulty-medium {{ color: #d69e2e; }}
        .prev-question .q-meta .difficulty-hard {{ color: #e53e3e; }}
        
        .no-questions {{ 
            color: #718096; 
            font-style: italic; 
            padding: 20px;
            text-align: center;
        }}
        
        .generated-section {{
            background: linear-gradient(135deg, #f6e05e 0%, #f6ad55 100%);
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .generated-section h2 {{ color: #744210; border-color: #d69e2e; }}
        
        .btn-generate {{
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            margin: 10px 0;
        }}
        .btn-generate:hover {{ opacity: 0.9; }}
    </style>
</head>
<body>
    <h1>🎯 AFCAT {year} Prediction Report</h1>
    
    <div class="summary-box">
        <strong>Generated:</strong> {generated_at}<br>
        <strong>Overall Confidence:</strong> {confidence:.1%}<br>
        <strong>Predicted Cutoff:</strong> {cutoff_low} - {cutoff_high}<br>
        <strong>Question Bank:</strong> {question_bank_status}
    </div>
    
    {sections_html}
    
    {generated_questions_html}
    
    <div class="section">
        <h2>📊 High-Yield Study Clusters</h2>
        <ol>
            {clusters_html}
        </ol>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; color: #718096;">
        <p>⚠️ This prediction is based on historical pattern analysis. Actual exam may vary.</p>
        <p>Use as preparation guide, not guaranteed predictions.</p>
        <p>💡 Click on any topic to see previous year questions from that topic.</p>
    </footer>
    
    <script>
        // Lazy load question data if needed
        document.querySelectorAll('.topic-dropdown').forEach(dropdown => {{
            dropdown.addEventListener('toggle', function() {{
                if (this.open && this.dataset.loaded !== 'true') {{
                    // Could fetch from JSON file here for large datasets
                    this.dataset.loaded = 'true';
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    
    # Prepare question bank data
    qb_data = question_bank_data or {{}}
    question_bank_status = f"{sum(len(qs) for qs in qb_data.values())} questions loaded" if qb_data else "Not loaded (run --build-bank first)"
    
    sections_html = ""
    for section, sec_pred in prediction.section_predictions.items():
        topics_html = ""
        for tp in sec_pred.topic_predictions[:8]:
            conf_class = 'confidence-high' if tp.confidence > 0.7 else ('confidence-medium' if tp.confidence > 0.5 else 'confidence-low')
            trend_class = f'trend-{tp.trend.value}'
            trend_arrow = '↑' if tp.trend == TrendDirection.RISING else ('↓' if tp.trend == TrendDirection.DECLINING else '→')
            
            # Build question list for this topic
            topic_questions = qb_data.get(tp.topic, [])
            questions_html = ""
            
            if topic_questions:
                for q in topic_questions[:10]:  # Limit to 10 questions per topic
                    year_str = f"AFCAT {q.get('year', '?')}" if q.get('year') else "Year Unknown"
                    shift_str = f"Shift {q.get('shift')}" if q.get('shift') else ""
                    diff = q.get('difficulty', 'medium')
                    diff_class = f"difficulty-{diff}"
                    
                    options_html = ""
                    for i, opt in enumerate(q.get('options', [])[:4]):
                        options_html += f"<div>({chr(65+i)}) {opt}</div>"
                    
                    questions_html += f"""
                    <div class="prev-question">
                        <div class="q-text">{q.get('text', q.get('question_text', 'Question text not available'))}</div>
                        <div class="options">{options_html}</div>
                        <div class="q-meta">
                            <span class="year">{year_str} {shift_str}</span>
                            <span class="{diff_class}">{diff.title()}</span>
                            <span>{q.get('question_type', 'Unknown Type')}</span>
                        </div>
                    </div>
                    """
            else:
                questions_html = '<div class="no-questions">No previous questions found for this topic. Run --build-bank to populate.</div>'
            
            # Create expandable topic row
            topics_html += f"""
            <details class="topic-dropdown" data-topic="{tp.topic}">
                <summary>
                    <span>{tp.topic.replace('_', ' ').title()}</span>
                    <div class="topic-meta">
                        <span>📊 {tp.predicted_count:.1f} Qs</span>
                        <span class="{conf_class}">{tp.confidence:.0%}</span>
                        <span class="{trend_class}">{trend_arrow}</span>
                        <span>📚 {len(topic_questions)} prev</span>
                    </div>
                </summary>
                <div class="questions-list">
                    <h4>Previous Year Questions ({len(topic_questions)} found)</h4>
                    {questions_html}
                </div>
            </details>
            """
            
        sections_html += f"""
        <div class="section">
            <h2>📚 {section.value.replace('_', ' ').title()}</h2>
            <p><strong>Questions:</strong> {sec_pred.total_questions} | 
               <strong>Difficulty:</strong> {sec_pred.predicted_difficulty.title()} |
               <strong>Good Attempts:</strong> {sec_pred.good_attempts_expected}</p>
            {topics_html}
        </div>
        """
    
    # Generated questions section (placeholder)
    generated_questions_html = """
    <div class="section generated-section">
        <h2>🤖 AI-Generated Predicted Questions</h2>
        <p>Generate predicted questions based on pattern analysis of previous years.</p>
        <p><em>Run with <code>--generate-questions</code> to create predicted questions using AI.</em></p>
    </div>
    """
        
    clusters_html = ""
    if prediction.study_plan.get('high_yield_clusters'):
        for topic in prediction.study_plan['high_yield_clusters'][:10]:
            clusters_html += f"<li>{topic.replace('_', ' ').title()}</li>\n"
            
    html = html_template.format(
        year=prediction.target_year,
        generated_at=prediction.generated_at.strftime('%Y-%m-%d %H:%M'),
        confidence=prediction.overall_confidence,
        cutoff_low=prediction.predicted_cutoff_range[0],
        cutoff_high=prediction.predicted_cutoff_range[1],
        sections_html=sections_html,
        clusters_html=clusters_html,
        question_bank_status=question_bank_status,
        generated_questions_html=generated_questions_html
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    logger.info(f"HTML report saved to {output_path}")
