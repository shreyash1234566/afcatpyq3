"""
Topic Analyzer Module
=====================
Analyzes historical topic patterns and generates insights.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_structures import (
    Question, Section, TopicFrequency, TopicPrediction,
    SectionPrediction, TrendDirection
)
from utils.feature_engine import build_topic_frequency_map, calculate_topic_roi

logger = logging.getLogger(__name__)


class TopicAnalyzer:
    """
    Analyzes historical topic frequencies and patterns.
    
    Provides insights for prediction models and study planning.
    """
    
    def __init__(self, questions: Optional[List[Question]] = None):
        self.questions = questions or []
        self.topic_frequencies: Dict[str, TopicFrequency] = {}
        self.section_stats: Dict[Section, Dict] = {}
        
        if questions:
            self._build_frequency_map()
            
    def _build_frequency_map(self):
        """Build topic frequency map from questions."""
        self.topic_frequencies = build_topic_frequency_map(self.questions)
        
        # Calculate section-level stats
        for section in Section:
            section_topics = {
                topic: freq for topic, freq in self.topic_frequencies.items()
                if freq.section == section
            }
            
            if section_topics:
                self.section_stats[section] = {
                    'total_topics': len(section_topics),
                    'avg_questions_per_topic': np.mean([
                        tf.average for tf in section_topics.values()
                    ]),
                    'most_common': max(section_topics.items(), key=lambda x: x[1].total_count)[0],
                    'rising_topics': [t for t, f in section_topics.items() if f.trend == TrendDirection.RISING],
                    'declining_topics': [t for t, f in section_topics.items() if f.trend == TrendDirection.DECLINING]
                }
                
    def load_from_config(self, historical_frequencies: Dict[str, List[int]], years: List[int]):
        """
        Load topic frequencies from config data.
        
        Args:
            historical_frequencies: {topic: [count_year1, count_year2, ...]}
            years: List of years corresponding to frequency indices
        """
        from config import TOPIC_TAXONOMY
        
        # Map topics to sections
        topic_to_section = {}
        for section, topics in TOPIC_TAXONOMY.items():
            section_enum = Section(section)
            for topic in topics.keys():
                topic_to_section[topic] = section_enum
                
        # Build TopicFrequency objects
        for topic, freq_list in historical_frequencies.items():
            frequencies = {year: count for year, count in zip(years, freq_list)}
            section = topic_to_section.get(topic, Section.GENERAL_AWARENESS)
            
            tf = TopicFrequency(
                topic=topic,
                section=section,
                frequencies=frequencies
            )
            tf.calculate_stats()
            self.topic_frequencies[topic] = tf
            
        logger.info(f"Loaded {len(self.topic_frequencies)} topics from config")
        
    def get_high_frequency_topics(
        self,
        section: Optional[Section] = None,
        top_n: int = 10
    ) -> List[Tuple[str, TopicFrequency]]:
        """Get topics with highest historical frequency."""
        topics = self.topic_frequencies.items()
        
        if section:
            topics = [(t, f) for t, f in topics if f.section == section]
            
        sorted_topics = sorted(topics, key=lambda x: x[1].average, reverse=True)
        return sorted_topics[:top_n]
    
    def get_trending_topics(
        self,
        direction: TrendDirection = TrendDirection.RISING,
        section: Optional[Section] = None
    ) -> List[Tuple[str, TopicFrequency]]:
        """Get topics with specific trend direction."""
        topics = []
        
        for topic, freq in self.topic_frequencies.items():
            if freq.trend == direction:
                if section is None or freq.section == section:
                    topics.append((topic, freq))
                    
        return sorted(topics, key=lambda x: abs(x[1].trend_coefficient), reverse=True)
    
    def get_dormant_topics(self, years_threshold: int = 2) -> List[Tuple[str, TopicFrequency]]:
        """Get topics that haven't appeared recently."""
        current_year = datetime.now().year
        dormant = []
        
        for topic, freq in self.topic_frequencies.items():
            if freq.last_appearance and (current_year - freq.last_appearance) >= years_threshold:
                dormant.append((topic, freq))
                
        return sorted(dormant, key=lambda x: x[1].last_appearance or 0)
    
    def predict_topic_count(
        self,
        topic: str,
        target_year: int,
        method: str = 'weighted_average'
    ) -> Tuple[float, float]:
        """
        Predict question count for a topic.
        
        Args:
            topic: Topic name
            target_year: Year to predict for
            method: Prediction method ('simple_average', 'weighted_average', 'trend_adjusted')
            
        Returns:
            (predicted_count, confidence)
        """
        if topic not in self.topic_frequencies:
            return 0.0, 0.0
            
        freq = self.topic_frequencies[topic]
        years = sorted(freq.frequencies.keys())
        values = [freq.frequencies[y] for y in years]
        
        if not values:
            return 0.0, 0.0
            
        if method == 'simple_average':
            prediction = np.mean(values)
            confidence = min(0.6 + len(values) * 0.05, 0.85)
            
        elif method == 'weighted_average':
            # Weight recent years more heavily
            weights = np.array([0.5 ** (len(values) - i - 1) for i in range(len(values))])
            weights = weights / weights.sum()
            prediction = np.average(values, weights=weights)
            confidence = min(0.65 + len(values) * 0.05, 0.90)
            
        elif method == 'trend_adjusted':
            # Base prediction on weighted average
            weights = np.array([0.5 ** (len(values) - i - 1) for i in range(len(values))])
            weights = weights / weights.sum()
            base = np.average(values, weights=weights)
            
            # Adjust for trend
            trend_adjustment = freq.trend_coefficient * base
            prediction = base + trend_adjustment
            
            # Higher confidence for stable trends
            trend_stability = 1 - min(abs(freq.trend_coefficient), 0.5)
            confidence = min(0.60 + len(values) * 0.05 + trend_stability * 0.1, 0.90)
            
        else:
            prediction = freq.average
            confidence = 0.5
            
        # Ensure non-negative
        prediction = max(0, prediction)
        
        return prediction, confidence
    
    def generate_section_analysis(self, section: Section) -> Dict:
        """Generate comprehensive analysis for a section."""
        section_topics = {
            t: f for t, f in self.topic_frequencies.items()
            if f.section == section
        }
        
        if not section_topics:
            return {"error": f"No data for section {section}"}
            
        # Get predictions for each topic
        predictions = []
        for topic, freq in section_topics.items():
            pred_count, confidence = self.predict_topic_count(topic, 2026, 'trend_adjusted')
            predictions.append({
                'topic': topic,
                'predicted_count': pred_count,
                'confidence': confidence,
                'trend': freq.trend.value,
                'historical_avg': freq.average,
                'consecutive_appearances': freq.consecutive_appearances
            })
            
        # Sort by predicted count
        predictions = sorted(predictions, key=lambda x: x['predicted_count'], reverse=True)
        
        # Calculate section totals
        total_predicted = sum(p['predicted_count'] for p in predictions)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        return {
            'section': section.value,
            'topic_count': len(section_topics),
            'total_questions_predicted': round(total_predicted, 1),
            'average_confidence': round(avg_confidence, 2),
            'rising_topics': [t for t, f in section_topics.items() if f.trend == TrendDirection.RISING],
            'declining_topics': [t for t, f in section_topics.items() if f.trend == TrendDirection.DECLINING],
            'predictions': predictions
        }
    
    def generate_full_analysis(self) -> Dict[Section, Dict]:
        """Generate analysis for all sections."""
        return {
            section: self.generate_section_analysis(section)
            for section in Section
        }
    
    def get_study_priorities(self, available_hours: float = 100) -> List[Dict]:
        """
        Generate prioritized study list based on ROI analysis.
        
        Args:
            available_hours: Total hours available for preparation
            
        Returns:
            List of topics with recommended study hours
        """
        priorities = []
        
        for topic, freq in self.topic_frequencies.items():
            pred_count, confidence = self.predict_topic_count(topic, 2026, 'trend_adjusted')
            
            # Estimate difficulty (0-1 scale)
            difficulty = 0.5  # Default medium
            if freq.section == Section.NUMERICAL_ABILITY:
                difficulty = 0.6
            elif freq.section == Section.GENERAL_AWARENESS:
                difficulty = 0.4
                
            # Calculate ROI
            roi = calculate_topic_roi(freq, difficulty)
            
            priorities.append({
                'topic': topic,
                'section': freq.section.value,
                'predicted_questions': round(pred_count, 1),
                'confidence': round(confidence, 2),
                'trend': freq.trend.value,
                'roi_score': round(roi, 2),
                'difficulty': 'high' if difficulty > 0.55 else ('low' if difficulty < 0.45 else 'medium')
            })
            
        # Sort by ROI
        priorities = sorted(priorities, key=lambda x: x['roi_score'], reverse=True)
        
        # Allocate hours proportionally to ROI
        total_roi = sum(p['roi_score'] for p in priorities if p['roi_score'] > 0)
        
        for p in priorities:
            if total_roi > 0 and p['roi_score'] > 0:
                p['recommended_hours'] = round(available_hours * (p['roi_score'] / total_roi), 1)
            else:
                p['recommended_hours'] = 0
                
        return priorities


def print_analysis_report(analysis: Dict[Section, Dict]):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("AFCAT 2026 TOPIC ANALYSIS REPORT")
    print("=" * 70)
    
    for section, data in analysis.items():
        print(f"\n{'─' * 70}")
        print(f"📚 {section.value.upper().replace('_', ' ')}")
        print(f"{'─' * 70}")
        
        print(f"   Total Topics: {data.get('topic_count', 0)}")
        print(f"   Predicted Questions: {data.get('total_questions_predicted', 0)}")
        print(f"   Average Confidence: {data.get('average_confidence', 0):.0%}")
        
        if data.get('rising_topics'):
            print(f"   📈 Rising: {', '.join(data['rising_topics'][:3])}")
        if data.get('declining_topics'):
            print(f"   📉 Declining: {', '.join(data['declining_topics'][:3])}")
            
        print("\n   Top Predicted Topics:")
        for i, pred in enumerate(data.get('predictions', [])[:5], 1):
            trend_icon = "📈" if pred['trend'] == 'rising' else ("📉" if pred['trend'] == 'declining' else "➡️")
            print(f"   {i}. {pred['topic']}: {pred['predicted_count']:.1f} questions "
                  f"({pred['confidence']:.0%} conf.) {trend_icon}")
                  
    print("\n" + "=" * 70)
