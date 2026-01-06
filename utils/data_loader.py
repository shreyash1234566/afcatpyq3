"""
Data Loader Utilities
=====================
Functions for loading and preprocessing AFCAT question data.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .data_structures import Question, Section, Difficulty

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and parsing of AFCAT question data."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.sample_dir = data_dir / "sample"
        
    def load_questions_from_json(self, filepath: Path) -> List[Question]:
        """Load questions from a JSON file."""
        questions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    try:
                        questions.append(Question.from_dict(item))
                    except Exception as e:
                        logger.warning(f"Failed to parse question: {e}")
            elif isinstance(data, dict) and "questions" in data:
                for item in data["questions"]:
                    try:
                        questions.append(Question.from_dict(item))
                    except Exception as e:
                        logger.warning(f"Failed to parse question: {e}")
                        
            logger.info(f"Loaded {len(questions)} questions from {filepath}")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
            
        return questions
    
    def load_questions_from_csv(self, filepath: Path) -> List[Question]:
        """Load questions from a CSV file."""
        questions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        question = Question(
                            id=row.get("id", f"q_{len(questions)}"),
                            year=int(row["year"]),
                            shift=int(row.get("shift", 1)),
                            section=Section(row["section"].lower().replace(" ", "_")),
                            topic=row["topic"],
                            subtopic=row.get("subtopic"),
                            text=row.get("text", ""),
                            difficulty=Difficulty(row.get("difficulty", "unknown").lower()),
                            source=row.get("source", "memory_based")
                        )
                        questions.append(question)
                    except Exception as e:
                        logger.warning(f"Failed to parse CSV row: {e}")
                        
            logger.info(f"Loaded {len(questions)} questions from {filepath}")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error reading CSV {filepath}: {e}")
            
        return questions
    
    def load_all_questions(self, directory: Optional[Path] = None) -> List[Question]:
        """Load all questions from a directory."""
        directory = directory or self.processed_dir
        all_questions = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return all_questions
            
        # Load JSON files
        for json_file in directory.glob("*.json"):
            all_questions.extend(self.load_questions_from_json(json_file))
            
        # Load CSV files
        for csv_file in directory.glob("*.csv"):
            all_questions.extend(self.load_questions_from_csv(csv_file))
            
        logger.info(f"Total questions loaded: {len(all_questions)}")
        return all_questions
    
    def save_questions_json(self, questions: List[Question], filepath: Path):
        """Save questions to a JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [q.to_dict() for q in questions]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(questions)} questions to {filepath}")
        
    def save_questions_csv(self, questions: List[Question], filepath: Path):
        """Save questions to a CSV file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not questions:
            logger.warning("No questions to save")
            return
            
        fieldnames = ["id", "year", "shift", "section", "topic", "subtopic", 
                     "text", "difficulty", "source"]
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in questions:
                writer.writerow({
                    "id": q.id,
                    "year": q.year,
                    "shift": q.shift,
                    "section": q.section.value,
                    "topic": q.topic,
                    "subtopic": q.subtopic or "",
                    "text": q.text,
                    "difficulty": q.difficulty.value,
                    "source": q.source
                })
        logger.info(f"Saved {len(questions)} questions to {filepath}")


def aggregate_by_year_topic(questions: List[Question]) -> Dict[int, Dict[str, int]]:
    """Aggregate questions by year and topic."""
    aggregated = {}
    
    for q in questions:
        if q.year not in aggregated:
            aggregated[q.year] = {}
        if q.topic not in aggregated[q.year]:
            aggregated[q.year][q.topic] = 0
        aggregated[q.year][q.topic] += 1
        
    return aggregated


def aggregate_by_section_topic(questions: List[Question]) -> Dict[Section, Dict[str, int]]:
    """Aggregate questions by section and topic."""
    aggregated = {}
    
    for q in questions:
        if q.section not in aggregated:
            aggregated[q.section] = {}
        if q.topic not in aggregated[q.section]:
            aggregated[q.section][q.topic] = 0
        aggregated[q.section][q.topic] += 1
        
    return aggregated


def filter_questions(
    questions: List[Question],
    years: Optional[List[int]] = None,
    sections: Optional[List[Section]] = None,
    topics: Optional[List[str]] = None,
    difficulty: Optional[List[Difficulty]] = None
) -> List[Question]:
    """Filter questions by various criteria."""
    filtered = questions
    
    if years:
        filtered = [q for q in filtered if q.year in years]
    if sections:
        filtered = [q for q in filtered if q.section in sections]
    if topics:
        filtered = [q for q in filtered if q.topic in topics]
    if difficulty:
        filtered = [q for q in filtered if q.difficulty in difficulty]
        
    return filtered
