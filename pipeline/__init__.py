# Pipeline module for AFCAT exam paper analysis
"""
Pipeline Package
================
End-to-end pipeline for extracting and analyzing AFCAT exam papers.

Usage:
    from pipeline import AFCATExamAnalyzer, analyze_exam_paper
    
    # Quick analysis
    result = analyze_exam_paper("afcat_2025.pdf", year=2025)
    
    # Full control
    analyzer = AFCATExamAnalyzer(use_transformers=True)
    result = analyzer.analyze_paper("afcat_2025.pdf", year=2025, shift=1)
"""

from .exam_analyzer import (
    AFCATExamAnalyzer,
    AnalyzedQuestion,
    PaperAnalysisResult,
    QualityChecker,
    QualityCheck,
    analyze_exam_paper,
    batch_analyze_papers,
    convert_to_prediction_format
)

__all__ = [
    "AFCATExamAnalyzer",
    "AnalyzedQuestion", 
    "PaperAnalysisResult",
    "QualityChecker",
    "QualityCheck",
    "analyze_exam_paper",
    "batch_analyze_papers",
    "convert_to_prediction_format"
]
