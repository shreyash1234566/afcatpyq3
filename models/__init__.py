"""
Models Package
==============
Machine learning models for AFCAT prediction.
"""

from .topic_predictor import (
    TopicPredictor,
    create_ensemble_predictor
)

from .difficulty_model import (
    DifficultyClassifier,
    estimate_paper_difficulty,
    get_normalization_advice
)

from .current_affairs import (
    CurrentAffairsClassifier,
    generate_current_affairs_summary,
    create_mock_news_data
)

__all__ = [
    # Topic prediction
    'TopicPredictor',
    'create_ensemble_predictor',
    
    # Difficulty classification
    'DifficultyClassifier',
    'estimate_paper_difficulty',
    'get_normalization_advice',
    
    # Current affairs
    'CurrentAffairsClassifier',
    'generate_current_affairs_summary',
    'create_mock_news_data'
]
