"""
Difficulty Classification Model
================================
Random Forest model for predicting question difficulty levels.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_structures import Question, Difficulty, Section

logger = logging.getLogger(__name__)


class DifficultyClassifier:
    """
    Classifies expected difficulty of AFCAT questions/sections.
    
    Uses Random Forest for multi-class classification.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'class_weight': class_weight,
            'random_state': random_state
        }
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = None
            
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        self.classes_ = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
        
    def extract_features(self, question: Question) -> List[float]:
        """
        Extract features from a question for difficulty classification.
        """
        text = question.text
        
        # Text-based features
        text_length = len(text)
        word_count = len(text.split())
        avg_word_length = text_length / max(word_count, 1)
        
        # Complexity indicators
        has_numbers = sum(1 for c in text if c.isdigit())
        has_math_symbols = sum(1 for c in text if c in '+-×÷=<>%')
        has_special_terms = sum(1 for term in ['ratio', 'percentage', 'profit', 'loss', 
                                                'speed', 'distance', 'time', 'work']
                               if term in text.lower())
        
        # Section encoding
        section_difficulty = {
            Section.NUMERICAL_ABILITY: 0.7,
            Section.REASONING: 0.6,
            Section.VERBAL_ABILITY: 0.4,
            Section.GENERAL_AWARENESS: 0.3
        }
        section_score = section_difficulty.get(question.section, 0.5)
        
        # Topic-based complexity
        hard_topics = ['simple_compound_interest', 'clocks', 'probability', 
                      'syllogism', 'reading_comprehension']
        easy_topics = ['synonyms', 'antonyms', 'defense', 'sports']
        
        topic_difficulty = 0.5
        if question.topic in hard_topics:
            topic_difficulty = 0.8
        elif question.topic in easy_topics:
            topic_difficulty = 0.3
            
        # Option analysis (if available)
        num_options = len(question.options)
        
        return [
            text_length / 100,  # Normalize
            word_count / 20,
            avg_word_length / 10,
            has_numbers / 5,
            has_math_symbols / 3,
            has_special_terms / 3,
            section_score,
            topic_difficulty,
            num_options / 4
        ]
    
    def fit(self, questions: List[Question]) -> Dict[str, float]:
        """
        Train the classifier on labeled questions.
        """
        if self.model is None:
            logger.warning("sklearn not available, using rule-based classification")
            self.is_fitted = True
            return {"status": "rule_based_mode"}
            
        # Filter questions with known difficulty
        labeled_questions = [q for q in questions if q.difficulty != Difficulty.UNKNOWN]
        
        if len(labeled_questions) < 10:
            logger.warning("Insufficient labeled data, using rule-based approach")
            self.is_fitted = True
            return {"status": "insufficient_data", "count": len(labeled_questions)}
            
        # Extract features
        X = np.array([self.extract_features(q) for q in labeled_questions])
        y = np.array([q.difficulty.value for q in labeled_questions])
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train model
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=3, scoring='f1_weighted')
        
        metrics = {
            'n_samples': len(labeled_questions),
            'cv_f1_mean': round(np.mean(cv_scores), 3),
            'cv_f1_std': round(np.std(cv_scores), 3),
            'classes': list(self.label_encoder.classes_)
        }
        
        logger.info(f"Difficulty classifier trained: F1={np.mean(cv_scores):.3f}")
        
        return metrics
    
    def predict(self, question: Question) -> Tuple[Difficulty, float]:
        """
        Predict difficulty for a single question.
        
        Returns:
            (predicted_difficulty, confidence)
        """
        if self.model is not None and self.is_fitted:
            features = np.array([self.extract_features(question)])
            
            # Get prediction and probabilities
            pred_encoded = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            confidence = float(max(proba))
            
            return Difficulty(pred_label), confidence
        else:
            # Rule-based fallback
            return self._rule_based_predict(question)
    
    def _rule_based_predict(self, question: Question) -> Tuple[Difficulty, float]:
        """
        Rule-based difficulty prediction when ML model unavailable.
        """
        # Topic-based rules
        hard_topics = {
            'simple_compound_interest', 'clocks', 'probability', 'mensuration',
            'syllogism', 'blood_relations', 'coding_decoding',
            'reading_comprehension', 'para_jumbles', 'economy', 'polity'
        }
        
        easy_topics = {
            'percentage', 'average', 'decimal_fractions',
            'synonyms', 'antonyms', 'idioms_phrases',
            'odd_one_out', 'direction_sense', 'defense', 'sports', 'static_gk'
        }
        
        # Section-based base difficulty
        section_base = {
            Section.NUMERICAL_ABILITY: Difficulty.MEDIUM,
            Section.REASONING: Difficulty.MEDIUM,
            Section.VERBAL_ABILITY: Difficulty.EASY,
            Section.GENERAL_AWARENESS: Difficulty.EASY
        }
        
        base_difficulty = section_base.get(question.section, Difficulty.MEDIUM)
        
        # Adjust based on topic
        if question.topic in hard_topics:
            return Difficulty.HARD, 0.7
        elif question.topic in easy_topics:
            return Difficulty.EASY, 0.7
        else:
            return base_difficulty, 0.5
    
    def predict_batch(self, questions: List[Question]) -> List[Tuple[Difficulty, float]]:
        """Predict difficulty for multiple questions."""
        return [self.predict(q) for q in questions]
    
    def predict_section_difficulty(
        self,
        section: Section,
        topic_distribution: Dict[str, int]
    ) -> Dict[str, any]:
        """
        Predict overall difficulty for a section based on topic distribution.
        """
        total_questions = sum(topic_distribution.values())
        
        if total_questions == 0:
            return {'difficulty': 'unknown', 'confidence': 0}
            
        # Calculate weighted difficulty
        hard_topics = {
            'simple_compound_interest', 'clocks', 'probability', 'syllogism',
            'reading_comprehension', 'blood_relations'
        }
        easy_topics = {
            'synonyms', 'antonyms', 'defense', 'sports', 'percentage', 'average'
        }
        
        hard_count = sum(count for topic, count in topic_distribution.items() 
                        if topic in hard_topics)
        easy_count = sum(count for topic, count in topic_distribution.items() 
                        if topic in easy_topics)
        medium_count = total_questions - hard_count - easy_count
        
        # Calculate difficulty score (0-1)
        difficulty_score = (hard_count * 1.0 + medium_count * 0.5 + easy_count * 0.0) / total_questions
        
        # Determine category
        if difficulty_score > 0.6:
            difficulty = 'difficult'
        elif difficulty_score > 0.4:
            difficulty = 'moderate'
        else:
            difficulty = 'easy'
            
        # Estimate good attempts
        good_attempts_ratio = {
            'easy': 0.85,
            'moderate': 0.70,
            'difficult': 0.55
        }
        good_attempts = int(total_questions * good_attempts_ratio[difficulty])
        
        return {
            'section': section.value,
            'difficulty': difficulty,
            'difficulty_score': round(difficulty_score, 2),
            'hard_topics_count': hard_count,
            'easy_topics_count': easy_count,
            'good_attempts_expected': good_attempts,
            'confidence': 0.65
        }


def estimate_paper_difficulty(
    section_predictions: Dict[Section, Dict[str, int]]
) -> Dict[str, any]:
    """
    Estimate overall paper difficulty from section topic distributions.
    """
    classifier = DifficultyClassifier()
    
    section_results = {}
    total_good_attempts = 0
    total_questions = 0
    
    for section, topic_dist in section_predictions.items():
        result = classifier.predict_section_difficulty(section, topic_dist)
        section_results[section.value] = result
        total_good_attempts += result['good_attempts_expected']
        total_questions += sum(topic_dist.values())
        
    # Overall paper difficulty
    overall_ratio = total_good_attempts / max(total_questions, 1)
    
    if overall_ratio > 0.75:
        overall_difficulty = 'easy'
    elif overall_ratio > 0.60:
        overall_difficulty = 'moderate'
    else:
        overall_difficulty = 'difficult'
        
    return {
        'overall_difficulty': overall_difficulty,
        'total_good_attempts': total_good_attempts,
        'total_questions': total_questions,
        'attempt_ratio': round(overall_ratio, 2),
        'sections': section_results,
        'normalization_advice': get_normalization_advice(overall_difficulty)
    }


def get_normalization_advice(difficulty: str) -> str:
    """
    Provide strategic advice based on predicted difficulty.
    
    AFCAT uses shift normalization - this helps candidates adjust strategy.
    """
    advice = {
        'easy': (
            "Expected EASY paper: Focus on maximizing attempts. "
            "Raw scores may normalize DOWN, so every question counts. "
            "Aim for 85+ attempts with 85%+ accuracy."
        ),
        'moderate': (
            "Expected MODERATE paper: Balance speed and accuracy. "
            "Normalization will be minimal. "
            "Aim for 75-80 attempts with 80%+ accuracy."
        ),
        'difficult': (
            "Expected DIFFICULT paper: Prioritize accuracy over attempts. "
            "Raw scores may normalize UP significantly. "
            "Aim for 65-70 attempts with 90%+ accuracy."
        )
    }
    return advice.get(difficulty, "Standard strategy: 75 attempts with 80% accuracy")
