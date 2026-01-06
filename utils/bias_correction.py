"""
Bias Correction Module
======================
Corrects for memory recall bias in historical data.
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging

from .data_structures import Question, Difficulty, Section

logger = logging.getLogger(__name__)


class BiasCorrector:
    """
    Corrects for cognitive biases in memory-based exam papers.
    
    Key biases addressed:
    1. Recall Bias: Candidates remember hard/unusual questions more
    2. Severity Bias: Difficult questions are over-reported
    3. Novelty Bias: Unique questions are remembered better
    """
    
    def __init__(
        self,
        expected_easy_ratio: float = 0.45,
        expected_medium_ratio: float = 0.35,
        expected_hard_ratio: float = 0.20,
        memory_recall_factor: float = 0.70
    ):
        self.expected_easy_ratio = expected_easy_ratio
        self.expected_medium_ratio = expected_medium_ratio
        self.expected_hard_ratio = expected_hard_ratio
        self.memory_recall_factor = memory_recall_factor
        
    def analyze_bias(self, questions: List[Question]) -> Dict[str, float]:
        """
        Analyze the bias present in the dataset.
        
        Returns metrics indicating level of bias.
        """
        if not questions:
            return {"error": "No questions to analyze"}
            
        # Count difficulty distribution
        difficulty_counts = defaultdict(int)
        for q in questions:
            difficulty_counts[q.difficulty] += 1
            
        total = len(questions)
        
        # Calculate observed ratios
        observed_easy = difficulty_counts[Difficulty.EASY] / total
        observed_medium = difficulty_counts[Difficulty.MEDIUM] / total
        observed_hard = difficulty_counts[Difficulty.HARD] / total
        observed_unknown = difficulty_counts[Difficulty.UNKNOWN] / total
        
        # Calculate bias metrics
        easy_bias = self.expected_easy_ratio - observed_easy
        hard_bias = observed_hard - self.expected_hard_ratio
        
        # Overall bias score (0 = no bias, 1 = severe bias)
        bias_score = (abs(easy_bias) + abs(hard_bias)) / 2
        
        return {
            "observed_easy_ratio": observed_easy,
            "observed_medium_ratio": observed_medium,
            "observed_hard_ratio": observed_hard,
            "observed_unknown_ratio": observed_unknown,
            "easy_underreporting": easy_bias,
            "hard_overreporting": hard_bias,
            "overall_bias_score": bias_score,
            "estimated_missing_questions": int(total * (1 - self.memory_recall_factor) / self.memory_recall_factor)
        }
    
    def correct_topic_frequencies(
        self,
        topic_frequencies: Dict[str, int],
        section: Section,
        expected_total: int
    ) -> Dict[str, float]:
        """
        Correct topic frequencies for recall bias.
        
        Args:
            topic_frequencies: Observed {topic: count}
            section: Which exam section
            expected_total: Expected total questions in section
            
        Returns:
            Corrected {topic: estimated_count}
        """
        observed_total = sum(topic_frequencies.values())
        
        if observed_total == 0:
            return topic_frequencies
            
        # Calculate missing questions
        recall_rate = self.memory_recall_factor
        estimated_actual = observed_total / recall_rate
        missing_count = estimated_actual - observed_total
        
        # Section-specific difficulty biases
        section_hard_topics = {
            Section.NUMERICAL_ABILITY: ["simple_compound_interest", "clocks", "probability"],
            Section.REASONING: ["syllogism", "blood_relations"],
            Section.VERBAL_ABILITY: ["para_jumbles", "reading_comprehension"],
            Section.GENERAL_AWARENESS: ["economy", "science"]
        }
        
        hard_topics = section_hard_topics.get(section, [])
        
        corrected = {}
        
        for topic, count in topic_frequencies.items():
            # Hard topics are over-reported, reduce their count
            if topic in hard_topics:
                correction_factor = 0.85  # Reduce by 15%
            else:
                # Easy/routine topics are under-reported, increase their count
                correction_factor = 1.15  # Increase by 15%
                
            corrected[topic] = count * correction_factor
            
        # Normalize to expected total
        corrected_sum = sum(corrected.values())
        if corrected_sum > 0:
            for topic in corrected:
                corrected[topic] = (corrected[topic] / corrected_sum) * expected_total
                
        return corrected
    
    def impute_missing_topics(
        self,
        topic_frequencies: Dict[str, int],
        all_topics: List[str],
        section: Section
    ) -> Dict[str, float]:
        """
        Impute likely counts for topics not appearing in memory-based data.
        
        Topics that appear in syllabus but not in memory data are likely
        easy/routine questions that were forgotten.
        """
        observed_topics = set(topic_frequencies.keys())
        missing_topics = set(all_topics) - observed_topics
        
        if not missing_topics:
            return dict(topic_frequencies)
            
        # Estimate count for missing topics
        # Assume they're easy questions that were forgotten
        avg_topic_count = np.mean(list(topic_frequencies.values())) if topic_frequencies else 1
        imputed_count = avg_topic_count * 0.5  # Assume half the average for missing
        
        result = dict(topic_frequencies)
        for topic in missing_topics:
            result[topic] = imputed_count
            
        logger.info(f"Imputed {len(missing_topics)} missing topics with estimated count {imputed_count:.1f}")
        
        return result
    
    def apply_full_correction(
        self,
        questions: List[Question],
        expected_totals: Dict[Section, int]
    ) -> Dict[Section, Dict[str, float]]:
        """
        Apply full bias correction pipeline.
        
        Returns corrected topic frequencies per section.
        """
        # Group questions by section and topic
        section_topics = defaultdict(lambda: defaultdict(int))
        
        for q in questions:
            section_topics[q.section][q.topic] += 1
            
        # Apply corrections
        corrected = {}
        
        for section, topic_freqs in section_topics.items():
            expected = expected_totals.get(section, sum(topic_freqs.values()))
            corrected[section] = self.correct_topic_frequencies(
                dict(topic_freqs),
                section,
                expected
            )
            
        # Log bias analysis
        bias_metrics = self.analyze_bias(questions)
        logger.info(f"Bias Analysis: {bias_metrics}")
        
        return corrected


def estimate_difficulty_from_topic(topic: str, section: Section) -> Difficulty:
    """
    Estimate difficulty level based on topic and section.
    
    Used when difficulty is not provided in source data.
    """
    hard_topics = {
        "simple_compound_interest", "clocks", "probability", "mensuration",
        "syllogism", "blood_relations", "coding_decoding",
        "reading_comprehension", "para_jumbles",
        "economy", "polity"
    }
    
    easy_topics = {
        "percentage", "average", "decimal_fractions",
        "synonyms", "antonyms", "idioms_phrases",
        "odd_one_out", "direction_sense",
        "defense", "sports", "static_gk"
    }
    
    if topic in hard_topics:
        return Difficulty.HARD
    elif topic in easy_topics:
        return Difficulty.EASY
    else:
        return Difficulty.MEDIUM
