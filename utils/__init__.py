"""
Utils Package
=============
Utility modules for AFCAT prediction system.
"""

from .data_structures import (
    Question,
    Section,
    Difficulty,
    TrendDirection,
    TopicFrequency,
    TopicPrediction,
    SectionPrediction,
    ExamPrediction,
    NewsArticle,
    MockTestBlueprint,
    StudyPlan
)

from .data_loader import (
    DataLoader,
    aggregate_by_year_topic,
    aggregate_by_section_topic,
    filter_questions
)

from .feature_engine import (
    FeatureEngine,
    build_topic_frequency_map,
    calculate_topic_roi
)

from .bias_correction import (
    BiasCorrector,
    estimate_difficulty_from_topic
)

__all__ = [
    # Data structures
    'Question',
    'Section',
    'Difficulty',
    'TrendDirection',
    'TopicFrequency',
    'TopicPrediction',
    'SectionPrediction',
    'ExamPrediction',
    'NewsArticle',
    'MockTestBlueprint',
    'StudyPlan',
    
    # Data loading
    'DataLoader',
    'aggregate_by_year_topic',
    'aggregate_by_section_topic',
    'filter_questions',
    
    # Feature engineering
    'FeatureEngine',
    'build_topic_frequency_map',
    'calculate_topic_roi',
    
    # Bias correction
    'BiasCorrector',
    'estimate_difficulty_from_topic'
]
