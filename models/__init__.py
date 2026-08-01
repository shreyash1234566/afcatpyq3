"""
Models Package
==============
Machine learning models for AFCAT prediction.
"""

from .difficulty_model import (
    DifficultyClassifier,
    estimate_paper_difficulty,
    get_normalization_advice
)

from .current_affairs import (
    CurrentAffairsClassifier,
    generate_current_affairs_summary
)

from .dirichlet_forecaster import DirichletForecaster

__all__ = [
    
    # Difficulty classification
    'DifficultyClassifier',
    'estimate_paper_difficulty',
    'get_normalization_advice',
    
    # Current affairs
    'CurrentAffairsClassifier',
    'generate_current_affairs_summary',
    'create_mock_news_data'
]
