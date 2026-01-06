"""
Data Structures for AFCAT Prediction System
============================================
Defines all data models used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import json


class Difficulty(Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNKNOWN = "unknown"


class Section(Enum):
    """AFCAT exam sections."""
    VERBAL_ABILITY = "verbal_ability"
    GENERAL_AWARENESS = "general_awareness"
    REASONING = "reasoning"
    NUMERICAL_ABILITY = "numerical_ability"


class TrendDirection(Enum):
    """Topic trend directions."""
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class Question:
    """Represents a single AFCAT question."""
    id: str
    year: int
    shift: int
    section: Section
    topic: str
    subtopic: Optional[str] = None
    text: str = ""
    options: List[str] = field(default_factory=list)
    correct_answer: Optional[str] = None
    difficulty: Difficulty = Difficulty.UNKNOWN
    source: str = "memory_based"  # 'official', 'memory_based', 'mock'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "year": self.year,
            "shift": self.shift,
            "section": self.section.value,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "text": self.text,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "difficulty": self.difficulty.value,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            id=data["id"],
            year=data["year"],
            shift=data["shift"],
            section=Section(data["section"]),
            topic=data["topic"],
            subtopic=data.get("subtopic"),
            text=data.get("text", ""),
            options=data.get("options", []),
            correct_answer=data.get("correct_answer"),
            difficulty=Difficulty(data.get("difficulty", "unknown")),
            source=data.get("source", "memory_based")
        )


@dataclass
class TopicFrequency:
    """Topic frequency data across years."""
    topic: str
    section: Section
    frequencies: Dict[int, int]  # {year: count}
    total_count: int = 0
    average: float = 0.0
    trend: TrendDirection = TrendDirection.STABLE
    trend_coefficient: float = 0.0
    last_appearance: Optional[int] = None
    consecutive_appearances: int = 0
    
    def calculate_stats(self):
        """Calculate derived statistics."""
        if self.frequencies:
            self.total_count = sum(self.frequencies.values())
            self.average = self.total_count / len(self.frequencies)
            years = sorted(self.frequencies.keys())
            self.last_appearance = years[-1] if years else None
            
            # Calculate trend
            if len(years) >= 3:
                recent = sum(self.frequencies.get(y, 0) for y in years[-2:]) / 2
                older = sum(self.frequencies.get(y, 0) for y in years[:-2]) / max(len(years) - 2, 1)
                if older > 0:
                    self.trend_coefficient = (recent - older) / older
                    if self.trend_coefficient > 0.15:
                        self.trend = TrendDirection.RISING
                    elif self.trend_coefficient < -0.15:
                        self.trend = TrendDirection.DECLINING
            
            # Calculate consecutive appearances
            for year in reversed(years):
                if self.frequencies.get(year, 0) > 0:
                    self.consecutive_appearances += 1
                else:
                    break


@dataclass
class TopicPrediction:
    """Prediction for a specific topic."""
    topic: str
    section: Section
    predicted_count: float
    confidence: float
    trend: TrendDirection
    priority_rank: int = 0
    study_hours_recommended: float = 0.0
    historical_average: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "section": self.section.value,
            "predicted_count": round(self.predicted_count, 1),
            "confidence": round(self.confidence, 2),
            "trend": self.trend.value,
            "priority_rank": self.priority_rank,
            "study_hours_recommended": round(self.study_hours_recommended, 1),
            "historical_average": round(self.historical_average, 1),
            "notes": self.notes
        }


@dataclass
class SectionPrediction:
    """Prediction for an entire section."""
    section: Section
    total_questions: int
    predicted_difficulty: str  # "easy", "moderate", "difficult"
    topic_predictions: List[TopicPrediction] = field(default_factory=list)
    good_attempts_expected: int = 0
    
    def get_high_priority_topics(self, top_n: int = 5) -> List[TopicPrediction]:
        """Get top N priority topics."""
        return sorted(self.topic_predictions, key=lambda x: x.priority_rank)[:top_n]


@dataclass
class ExamPrediction:
    """Complete prediction for AFCAT 2026."""
    target_year: int
    generated_at: datetime
    section_predictions: Dict[Section, SectionPrediction] = field(default_factory=dict)
    overall_confidence: float = 0.0
    predicted_cutoff_range: tuple = (155, 165)
    study_plan: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_year": self.target_year,
            "generated_at": self.generated_at.isoformat(),
            "overall_confidence": round(self.overall_confidence, 2),
            "predicted_cutoff_range": list(self.predicted_cutoff_range),
            "sections": {
                section.value: {
                    "total_questions": pred.total_questions,
                    "predicted_difficulty": pred.predicted_difficulty,
                    "good_attempts_expected": pred.good_attempts_expected,
                    "topics": [tp.to_dict() for tp in pred.topic_predictions]
                }
                for section, pred in self.section_predictions.items()
            },
            "study_plan": self.study_plan
        }
    
    def save_json(self, filepath: str):
        """Save prediction to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class NewsArticle:
    """News article for current affairs analysis."""
    title: str
    content: str
    source: str
    published_date: datetime
    url: str = ""
    category: str = ""
    relevance_score: float = 0.0
    afcat_probability: float = 0.0
    key_facts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "url": self.url,
            "category": self.category,
            "relevance_score": round(self.relevance_score, 2),
            "afcat_probability": round(self.afcat_probability, 2),
            "key_facts": self.key_facts
        }


@dataclass
class MockTestBlueprint:
    """Blueprint for generating mock tests."""
    name: str
    based_on_year: int
    sections: Dict[Section, Dict[str, int]]  # {section: {topic: count}}
    total_questions: int = 100
    difficulty_distribution: Dict[Difficulty, float] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate blueprint has correct total questions."""
        total = sum(
            sum(topics.values()) 
            for topics in self.sections.values()
        )
        return total == self.total_questions


@dataclass
class StudyPlan:
    """Generated study plan based on predictions."""
    target_exam: str
    generated_date: datetime
    days_until_exam: int
    priority_topics: List[Dict[str, Any]]
    daily_schedule: Dict[str, List[Dict[str, Any]]]
    weekly_goals: List[str]
    high_yield_clusters: List[str]
    revision_schedule: Dict[str, List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_exam": self.target_exam,
            "generated_date": self.generated_date.isoformat(),
            "days_until_exam": self.days_until_exam,
            "priority_topics": self.priority_topics,
            "daily_schedule": self.daily_schedule,
            "weekly_goals": self.weekly_goals,
            "high_yield_clusters": self.high_yield_clusters,
            "revision_schedule": self.revision_schedule
        }
