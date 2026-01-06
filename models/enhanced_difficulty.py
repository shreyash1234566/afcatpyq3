"""
Enhanced Difficulty Predictor
=============================
Predicts question difficulty (easy/medium/hard) using:
- Text features (length, complexity, vocabulary)
- Mathematical complexity
- Option similarity analysis
- Topic-based baseline difficulty
"""

from typing import List, Dict, Tuple, Optional
import re
import math
from dataclasses import dataclass


@dataclass
class DifficultyFeatures:
    """Features extracted for difficulty prediction."""
    text_length: int
    word_count: int
    avg_word_length: float
    sentence_count: int
    numeric_count: int
    operator_count: int
    has_multiple_steps: bool
    vocabulary_complexity: float
    option_similarity: float
    topic_base_difficulty: float
    has_formula: bool
    has_diagram_reference: bool


class EnhancedDifficultyPredictor:
    """
    Predicts question difficulty using multiple text and context features.
    """
    
    # Topic-wise base difficulty (from historical AFCAT analysis)
    # Scale: 0.0 (very easy) to 1.0 (very hard)
    TOPIC_DIFFICULTY = {
        # Numerical - Hard topics (0.7-1.0)
        "simple_compound_interest": 0.8,
        "probability": 0.85,
        "permutation_combination": 0.9,
        "data_interpretation": 0.8,
        "mensuration": 0.7,
        "algebra": 0.65,
        
        # Numerical - Medium topics (0.4-0.7)
        "speed_time_distance": 0.6,
        "time_and_work": 0.55,
        "profit_loss": 0.5,
        "geometry": 0.6,
        "number_series": 0.5,
        "mixtures_alligation": 0.65,
        
        # Numerical - Easy topics (0.1-0.4)
        "percentages": 0.4,
        "ratio_proportion": 0.4,
        "averages": 0.35,
        "simplification": 0.3,
        "number_system": 0.35,
        "lcm_hcf": 0.35,
        
        # Verbal - Hard
        "reading_comprehension": 0.7,
        "cloze_test": 0.65,
        "para_jumbles": 0.7,
        
        # Verbal - Medium
        "idioms_phrases": 0.5,
        "one_word_substitution": 0.5,
        "error_detection": 0.55,
        "sentence_improvement": 0.5,
        "direct_indirect": 0.55,
        "active_passive": 0.45,
        
        # Verbal - Easy
        "synonyms": 0.25,
        "antonyms": 0.2,
        "spelling": 0.2,
        "sentence_completion": 0.4,
        "vocabulary": 0.35,
        
        # Reasoning - Hard
        "syllogism": 0.75,
        "seating_arrangement": 0.8,
        "puzzle": 0.8,
        "cubes_dice": 0.7,
        "spatial_ability": 0.7,
        
        # Reasoning - Medium
        "coding_decoding": 0.55,
        "blood_relations": 0.55,
        "direction_sense": 0.5,
        "venn_diagrams": 0.5,
        "series_completion": 0.5,
        "paper_folding": 0.6,
        "embedded_figures": 0.55,
        
        # Reasoning - Easy
        "analogy": 0.35,
        "odd_one_out": 0.3,
        "mirror_water_images": 0.4,
        "calendar": 0.45,
        "clocks": 0.5,
        "ranking_ordering": 0.4,
        
        # GK - Generally medium (depends on preparation)
        "defense": 0.4,
        "history": 0.5,
        "geography": 0.45,
        "polity": 0.55,
        "economy": 0.6,
        "science": 0.5,
        "current_affairs": 0.5,
        "sports": 0.35,
        "awards": 0.35,
        "books_authors": 0.45,
        "important_days": 0.3,
        "environment": 0.5,
        "technology": 0.5,
        "art_culture": 0.5,
    }
    
    # Complex vocabulary indicators
    COMPLEX_WORDS = {
        "notwithstanding", "henceforth", "whereas", "thereby",
        "subsequently", "consequently", "nevertheless", "furthermore",
        "aforementioned", "heretofore", "thereupon", "albeit",
        "ergo", "hence", "thus", "moreover", "paradoxically",
        "ostensibly", "presumably", "inherently", "intrinsically"
    }
    
    # Multi-step indicators
    MULTI_STEP_INDICATORS = [
        "then", "after", "next", "finally", "first", "second",
        "step", "followed by", "subsequently", "afterwards",
        "if...then", "and then"
    ]
    
    def __init__(self):
        pass
    
    def extract_features(
        self,
        question_text: str,
        options: List[str] = None,
        topic: str = None
    ) -> DifficultyFeatures:
        """Extract all difficulty-related features from question."""
        
        text = question_text
        text_lower = text.lower()
        words = text.split()
        
        # Basic text features
        text_length = len(text)
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        
        # Sentence complexity
        sentences = re.split(r'[.!?]', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Mathematical complexity
        numeric_count = len(re.findall(r'\d+', text))
        operator_count = len(re.findall(r'[+\-×÷*/^%=<>]', text))
        
        # Multi-step detection
        has_multiple_steps = any(
            ind in text_lower for ind in self.MULTI_STEP_INDICATORS
        )
        
        # Vocabulary complexity
        complex_word_count = sum(
            1 for word in words 
            if word.lower().strip('.,!?;:') in self.COMPLEX_WORDS
        )
        vocab_complexity = complex_word_count / max(word_count, 1)
        
        # Option similarity (if provided)
        option_similarity = 0.0
        if options and len(options) >= 2:
            option_similarity = self._calculate_option_similarity(options)
            
        # Topic base difficulty
        topic_difficulty = self.TOPIC_DIFFICULTY.get(topic, 0.5) if topic else 0.5
        
        # Formula detection
        has_formula = bool(re.search(
            r'\d+\s*[+\-×÷*/^]\s*\d+|[a-z]\s*=|√|π|\^2|\^3',
            text, re.I
        ))
        
        # Diagram/figure reference
        has_diagram_reference = bool(re.search(
            r'figure|diagram|table|graph|chart|above|below|given',
            text, re.I
        ))
        
        return DifficultyFeatures(
            text_length=text_length,
            word_count=word_count,
            avg_word_length=avg_word_length,
            sentence_count=sentence_count,
            numeric_count=numeric_count,
            operator_count=operator_count,
            has_multiple_steps=has_multiple_steps,
            vocabulary_complexity=vocab_complexity,
            option_similarity=option_similarity,
            topic_base_difficulty=topic_difficulty,
            has_formula=has_formula,
            has_diagram_reference=has_diagram_reference
        )
    
    def _calculate_option_similarity(self, options: List[str]) -> float:
        """
        Calculate how similar options are.
        More similar options = harder question (tricky options).
        """
        if len(options) < 2:
            return 0.0
        
        # Clean options
        clean_options = [opt.lower().strip() for opt in options]
        
        similarities = []
        for i in range(len(clean_options)):
            for j in range(i + 1, len(clean_options)):
                # Character set similarity (Jaccard)
                set1 = set(clean_options[i])
                set2 = set(clean_options[j])
                
                if not set1 or not set2:
                    continue
                    
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                if union > 0:
                    similarities.append(intersection / union)
                    
                # Check for numeric closeness
                nums1 = re.findall(r'\d+', clean_options[i])
                nums2 = re.findall(r'\d+', clean_options[j])
                
                if nums1 and nums2:
                    try:
                        n1, n2 = int(nums1[0]), int(nums2[0])
                        # Close numbers are tricky
                        if abs(n1 - n2) <= max(n1, n2) * 0.2:
                            similarities.append(0.8)
                    except ValueError:
                        pass
                        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def predict(
        self,
        question_text: str,
        options: List[str] = None,
        topic: str = None
    ) -> Tuple[str, float, Dict]:
        """
        Predict difficulty level.
        
        Args:
            question_text: The question text
            options: List of answer options
            topic: The classified topic
            
        Returns:
            Tuple of (difficulty_level, confidence, feature_breakdown)
            Levels: "easy", "medium", "hard"
        """
        features = self.extract_features(question_text, options, topic)
        
        # Calculate score components
        scores = {}
        
        # 1. Text complexity score (0-0.2)
        text_score = 0.0
        if features.word_count > 50:
            text_score = 0.2
        elif features.word_count > 30:
            text_score = 0.15
        elif features.word_count > 15:
            text_score = 0.1
        else:
            text_score = 0.05
        scores['text_complexity'] = text_score
        
        # 2. Mathematical complexity score (0-0.25)
        math_score = 0.0
        if features.operator_count >= 4:
            math_score = 0.25
        elif features.operator_count >= 2:
            math_score = 0.15
        elif features.operator_count >= 1:
            math_score = 0.1
            
        if features.numeric_count >= 6:
            math_score += 0.1
        elif features.numeric_count >= 3:
            math_score += 0.05
            
        math_score = min(math_score, 0.25)
        scores['mathematical'] = math_score
        
        # 3. Multi-step score (0-0.15)
        multi_step_score = 0.15 if features.has_multiple_steps else 0.0
        scores['multi_step'] = multi_step_score
        
        # 4. Option difficulty (0-0.15)
        option_score = 0.0
        if features.option_similarity > 0.7:
            option_score = 0.15
        elif features.option_similarity > 0.5:
            option_score = 0.1
        elif features.option_similarity > 0.3:
            option_score = 0.05
        scores['option_similarity'] = option_score
        
        # 5. Topic baseline (0-0.35) - weighted heavily
        topic_score = features.topic_base_difficulty * 0.35
        scores['topic_baseline'] = topic_score
        
        # 6. Vocabulary complexity (0-0.1)
        vocab_score = min(features.vocabulary_complexity * 2, 0.1)
        scores['vocabulary'] = vocab_score
        
        # Total score
        total_score = sum(scores.values())
        
        # Normalize to 0-1
        final_score = min(max(total_score, 0), 1)
        
        # Determine level
        if final_score < 0.35:
            level = "easy"
        elif final_score < 0.6:
            level = "medium"
        else:
            level = "hard"
            
        # Calculate confidence based on how clear the classification is
        if final_score < 0.25 or final_score > 0.7:
            # Clear easy or hard
            confidence = 0.85
        elif final_score < 0.35 or final_score > 0.6:
            # Near boundaries but clear
            confidence = 0.75
        else:
            # Medium range - less certain
            confidence = 0.65
            
        return (level, confidence, {
            'score': final_score,
            'components': scores,
            'features': {
                'word_count': features.word_count,
                'numeric_count': features.numeric_count,
                'has_formula': features.has_formula,
                'has_multiple_steps': features.has_multiple_steps,
                'option_similarity': features.option_similarity
            }
        })
    
    def predict_batch(
        self,
        questions: List[Dict]
    ) -> List[Tuple[str, float]]:
        """
        Predict difficulty for multiple questions.
        
        Args:
            questions: List of dicts with 'text', 'options', 'topic' keys
        """
        results = []
        
        for q in questions:
            level, conf, _ = self.predict(
                q.get('text', ''),
                q.get('options'),
                q.get('topic')
            )
            results.append((level, conf))
            
        return results
    
    def get_difficulty_stats(
        self,
        predictions: List[Tuple[str, float]]
    ) -> Dict:
        """Get statistics from difficulty predictions."""
        counts = {"easy": 0, "medium": 0, "hard": 0}
        confidences = {"easy": [], "medium": [], "hard": []}
        
        for level, conf in predictions:
            counts[level] = counts.get(level, 0) + 1
            confidences[level].append(conf)
            
        total = sum(counts.values())
        
        stats = {
            "total": total,
            "distribution": {
                k: {"count": v, "percentage": v / total * 100 if total else 0}
                for k, v in counts.items()
            },
            "avg_confidence": {
                k: sum(v) / len(v) if v else 0
                for k, v in confidences.items()
            }
        }
        
        return stats


class DifficultyCalibrator:
    """
    Calibrates difficulty predictions based on actual exam results.
    Can be trained with historical data.
    """
    
    def __init__(self):
        self.calibration_factors = {
            "easy": 1.0,
            "medium": 1.0,
            "hard": 1.0
        }
        self.topic_adjustments = {}
        
    def calibrate_with_results(
        self,
        predictions: List[str],
        actual_difficulty: List[str]
    ):
        """
        Adjust calibration based on actual vs predicted.
        
        Args:
            predictions: List of predicted difficulty levels
            actual_difficulty: List of actual difficulty levels
        """
        # Count mismatches
        errors = {"easy": 0, "medium": 0, "hard": 0}
        totals = {"easy": 0, "medium": 0, "hard": 0}
        
        for pred, actual in zip(predictions, actual_difficulty):
            totals[pred] = totals.get(pred, 0) + 1
            if pred != actual:
                errors[pred] = errors.get(pred, 0) + 1
                
        # Adjust factors
        for level in ["easy", "medium", "hard"]:
            if totals[level] > 0:
                error_rate = errors[level] / totals[level]
                # Reduce confidence for high error rates
                self.calibration_factors[level] = 1.0 - (error_rate * 0.5)
                
    def apply_calibration(
        self,
        level: str,
        confidence: float
    ) -> float:
        """Apply calibration factor to confidence."""
        factor = self.calibration_factors.get(level, 1.0)
        return confidence * factor
