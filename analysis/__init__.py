"""
Analysis Package
================
Topic analysis and trend detection modules.
"""

from .topic_analyzer import TopicAnalyzer, print_analysis_report
from .trend_detector import (
    TrendDetector,
    identify_afcat_2024_break,
    get_hot_topics,
    get_cold_topics
)

__all__ = [
    'TopicAnalyzer',
    'print_analysis_report',
    'TrendDetector',
    'identify_afcat_2024_break',
    'get_hot_topics',
    'get_cold_topics'
]
