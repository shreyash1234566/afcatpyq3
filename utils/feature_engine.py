"""
Feature Engineering Module
==========================
Creates features for ML models from raw question data.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from .data_structures import Question, Section, TopicFrequency, TrendDirection

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Creates features for topic prediction models."""
    
    def __init__(self, lookback_years: int = 5):
        self.lookback_years = lookback_years
        
    def create_topic_features(
        self,
        topic_frequencies: Dict[str, TopicFrequency],
        target_year: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for topic prediction.
        
        Returns:
            features: numpy array of shape (n_topics, n_features)
            feature_names: list of feature names
        """
        feature_names = [
            'avg_frequency',
            'recent_frequency',
            'trend_coefficient',
            'years_since_last',
            'consecutive_appearances',
            'volatility',
            'max_frequency',
            'min_frequency',
            'frequency_year_1',
            'frequency_year_2',
            'frequency_year_3',
            'frequency_year_4',
            'frequency_year_5'
        ]
        
        features = []
        
        for topic, freq_data in topic_frequencies.items():
            freq_data.calculate_stats()
            
            # Get frequency values for lookback years
            years = sorted(freq_data.frequencies.keys())
            recent_years = [y for y in years if y >= target_year - self.lookback_years]
            freq_values = [freq_data.frequencies.get(y, 0) for y in recent_years]
            
            # Calculate features
            avg_freq = np.mean(freq_values) if freq_values else 0
            recent_freq = np.mean(freq_values[-2:]) if len(freq_values) >= 2 else avg_freq
            
            years_since_last = target_year - freq_data.last_appearance if freq_data.last_appearance else 10
            volatility = np.std(freq_values) if len(freq_values) > 1 else 0
            max_freq = max(freq_values) if freq_values else 0
            min_freq = min(freq_values) if freq_values else 0
            
            # Pad frequency history to 5 years
            padded_freq = [0] * 5
            for i, val in enumerate(freq_values[-5:]):
                padded_freq[i] = val
                
            topic_features = [
                avg_freq,
                recent_freq,
                freq_data.trend_coefficient,
                years_since_last,
                freq_data.consecutive_appearances,
                volatility,
                max_freq,
                min_freq,
                *padded_freq
            ]
            
            features.append(topic_features)
            
        return np.array(features), feature_names
    
    def create_difficulty_features(self, questions: List[Question]) -> Tuple[np.ndarray, List[str]]:
        """
        Create features for difficulty classification.
        
        Features based on question text characteristics.
        """
        feature_names = [
            'text_length',
            'word_count',
            'avg_word_length',
            'has_numbers',
            'has_formula_indicators',
            'option_count',
            'question_complexity_score'
        ]
        
        features = []
        
        for q in questions:
            text = q.text
            words = text.split() if text else []
            
            text_length = len(text)
            word_count = len(words)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            has_numbers = 1 if any(c.isdigit() for c in text) else 0
            has_formula = 1 if any(ind in text.lower() for ind in ['%', 'ratio', 'rs.', '₹', 'km', 'hr']) else 0
            option_count = len(q.options)
            
            # Complexity score based on section
            complexity_scores = {
                Section.NUMERICAL_ABILITY: 0.7,
                Section.REASONING: 0.6,
                Section.VERBAL_ABILITY: 0.4,
                Section.GENERAL_AWARENESS: 0.3
            }
            complexity = complexity_scores.get(q.section, 0.5)
            
            features.append([
                text_length,
                word_count,
                avg_word_length,
                has_numbers,
                has_formula,
                option_count,
                complexity
            ])
            
        return np.array(features), feature_names
    
    def create_temporal_features(
        self,
        historical_data: Dict[int, Dict[str, int]],
        target_year: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Create temporal features for each topic.
        
        Args:
            historical_data: {year: {topic: count}}
            target_year: Year to predict for
            
        Returns:
            Dict mapping topics to their temporal features
        """
        all_topics = set()
        for year_data in historical_data.values():
            all_topics.update(year_data.keys())
            
        temporal_features = {}
        years = sorted(historical_data.keys())
        
        for topic in all_topics:
            # Get time series for this topic
            series = [historical_data.get(y, {}).get(topic, 0) for y in years]
            
            # Calculate temporal features
            if len(series) >= 3:
                # Simple linear trend
                x = np.arange(len(series))
                coeffs = np.polyfit(x, series, 1)
                trend_slope = coeffs[0]
                
                # Moving average (last 3 years)
                ma_3 = np.mean(series[-3:])
                
                # Exponential smoothing prediction
                alpha = 0.3
                ema = series[0]
                for val in series[1:]:
                    ema = alpha * val + (1 - alpha) * ema
                    
                # Seasonality check (AFCAT-1 vs AFCAT-2 pattern)
                # Assuming alternating pattern
                even_years = [series[i] for i in range(0, len(series), 2)]
                odd_years = [series[i] for i in range(1, len(series), 2)]
                seasonality = abs(np.mean(even_years) - np.mean(odd_years)) if odd_years else 0
                
            else:
                trend_slope = 0
                ma_3 = np.mean(series) if series else 0
                ema = series[-1] if series else 0
                seasonality = 0
                
            temporal_features[topic] = {
                'trend_slope': trend_slope,
                'moving_average_3': ma_3,
                'ema_prediction': ema,
                'seasonality_score': seasonality,
                'last_value': series[-1] if series else 0,
                'years_of_data': len(series)
            }
            
        return temporal_features


def build_topic_frequency_map(
    questions: List[Question]
) -> Dict[str, TopicFrequency]:
    """
    Build topic frequency map from questions.
    
    Returns:
        Dict mapping topic names to TopicFrequency objects
    """
    # Group by topic
    topic_data = defaultdict(lambda: {"section": None, "frequencies": defaultdict(int)})
    
    for q in questions:
        topic_data[q.topic]["section"] = q.section
        topic_data[q.topic]["frequencies"][q.year] += 1
        
    # Create TopicFrequency objects
    topic_frequencies = {}
    for topic, data in topic_data.items():
        tf = TopicFrequency(
            topic=topic,
            section=data["section"],
            frequencies=dict(data["frequencies"])
        )
        tf.calculate_stats()
        topic_frequencies[topic] = tf
        
    return topic_frequencies


def calculate_topic_roi(
    topic_freq: TopicFrequency,
    avg_difficulty: float = 0.5,
    preparation_time: float = 1.0
) -> float:
    """
    Calculate Return on Investment for studying a topic.
    
    ROI = (Expected Questions * Probability of Correct) / Preparation Time
    
    Higher ROI = Higher priority for study
    """
    expected_questions = topic_freq.average
    
    # Adjust for trend
    trend_multiplier = {
        TrendDirection.RISING: 1.2,
        TrendDirection.STABLE: 1.0,
        TrendDirection.DECLINING: 0.8
    }
    expected_questions *= trend_multiplier.get(topic_freq.trend, 1.0)
    
    # Probability of correct answer (inverse of difficulty)
    prob_correct = 1 - avg_difficulty
    
    # Marks per question (3 marks, -1 for wrong)
    expected_marks = expected_questions * (3 * prob_correct - 1 * (1 - prob_correct))
    
    # ROI = expected marks / preparation time
    roi = expected_marks / max(preparation_time, 0.1)
    
    return roi
