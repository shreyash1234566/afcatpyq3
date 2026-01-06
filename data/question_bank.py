"""
Question Bank Database
======================
SQLite-based storage for all extracted AFCAT questions.
Enables fast queries by topic, year, difficulty, and question type.
Supports the Question Bank dropdown feature and Question Generator RAG retrieval.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class StoredQuestion:
    """A question stored in the database."""
    id: int
    question_text: str
    options: List[str]
    correct_answer: Optional[str]
    section: str
    topic: str
    subtopic: Optional[str]
    question_type: str
    difficulty: str
    year: Optional[int]
    shift: Optional[int]
    source_file: str
    topic_confidence: float
    difficulty_confidence: float
    has_formula: bool
    has_diagram: bool
    image_refs: List[Dict]
    created_at: str
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'StoredQuestion':
        """Create from database row."""
        return cls(
            id=row['id'],
            question_text=row['question_text'],
            options=json.loads(row['options']) if row['options'] else [],
            correct_answer=row['correct_answer'],
            section=row['section'],
            topic=row['topic'],
            subtopic=row['subtopic'],
            question_type=row['question_type'],
            difficulty=row['difficulty'],
            year=row['year'],
            shift=row['shift'],
            source_file=row['source_file'],
            topic_confidence=row['topic_confidence'],
            difficulty_confidence=row['difficulty_confidence'],
            has_formula=bool(row['has_formula']),
            has_diagram=bool(row['has_diagram']),
            image_refs=json.loads(row['image_refs']) if row['image_refs'] else [],
            created_at=row['created_at']
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class QuestionBankDB:
    """
    SQLite database for storing and querying AFCAT questions.
    
    Features:
    - Store questions with full metadata
    - Query by topic, year, difficulty, question type
    - Retrieve similar questions for RAG
    - Get topic statistics across years
    - Export to JSON for HTML report
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        if db_path is None:
            db_path = Path(__file__).parent / "question_bank.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"Question bank initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main questions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_text TEXT NOT NULL,
                    question_hash TEXT UNIQUE,
                    options TEXT,
                    correct_answer TEXT,
                    section TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    subtopic TEXT,
                    question_type TEXT,
                    difficulty TEXT,
                    year INTEGER,
                    shift INTEGER,
                    source_file TEXT,
                    topic_confidence REAL,
                    difficulty_confidence REAL,
                    has_formula INTEGER DEFAULT 0,
                    has_diagram INTEGER DEFAULT 0,
                    image_refs TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for fast queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic ON questions(topic)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_section ON questions(section)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON questions(year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_difficulty ON questions(difficulty)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question_type ON questions(question_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_year ON questions(topic, year)')
            
            # Topic statistics cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS topic_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    year INTEGER,
                    question_count INTEGER,
                    avg_difficulty REAL,
                    common_question_types TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(topic, year)
                )
            ''')
            
            logger.info("Database tables initialized")

            # Ensure new columns exist for backward compatibility
            self._ensure_column(cursor, 'questions', 'image_refs', 'TEXT')
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        import hashlib
        # Normalize text before hashing
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _ensure_column(self, cursor, table: str, column: str, definition: str):
        """Add a column if it does not already exist."""
        cursor.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cursor.fetchall()}
        if column not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    
    def add_question(
        self,
        question_text: str,
        options: List[str],
        section: str,
        topic: str,
        difficulty: str,
        question_type: str = "unknown",
        subtopic: Optional[str] = None,
        correct_answer: Optional[str] = None,
        year: Optional[int] = None,
        shift: Optional[int] = None,
        source_file: str = "",
        topic_confidence: float = 0.0,
        difficulty_confidence: float = 0.0,
        has_formula: bool = False,
        has_diagram: bool = False,
        image_refs: Optional[List[Dict]] = None
    ) -> Optional[int]:
        """
        Add a question to the database.
        Returns question ID if added, None if duplicate.
        """
        question_hash = self._compute_hash(question_text)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO questions (
                        question_text, question_hash, options, correct_answer,
                        section, topic, subtopic, question_type, difficulty,
                        year, shift, source_file, topic_confidence, 
                        difficulty_confidence, has_formula, has_diagram
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    question_text,
                    question_hash,
                    json.dumps(options),
                    correct_answer,
                    section,
                    topic,
                    subtopic,
                    question_type,
                    difficulty,
                    year,
                    shift,
                    source_file,
                    topic_confidence,
                    difficulty_confidence,
                    int(has_formula),
                    int(has_diagram)
                ))
                
                question_id = cursor.lastrowid
                logger.debug(f"Added question {question_id}: {topic}")
                return question_id
                
            except sqlite3.IntegrityError:
                # Duplicate question
                logger.debug(f"Duplicate question skipped: {question_text[:50]}...")
                return None
        """
        Import questions from pipeline analysis result.
        Returns (added_count, duplicate_count).
        """
        added = 0
        duplicates = 0
        
        for q in analysis_result.get('questions', []):
            result = self.add_question(
                question_text=q.get('text', ''),
                options=q.get('options', []),
                section=q.get('section', 'unknown'),
                topic=q.get('topic', 'unknown'),
                subtopic=q.get('subtopic'),
                question_type=q.get('question_type', 'unknown'),
                difficulty=q.get('difficulty', 'medium'),
                year=analysis_result.get('year'),
                shift=analysis_result.get('shift'),
                source_file=analysis_result.get('source_file', ''),
                topic_confidence=q.get('confidence_scores', {}).get('topic', 0.0),
                difficulty_confidence=q.get('confidence_scores', {}).get('difficulty', 0.0),
                has_formula=q.get('has_formula', False),
                    has_diagram=q.get('has_diagram_reference', False),
                    image_refs=q.get('image_refs', [])
            )
            
            if result:
                added += 1
            else:
                duplicates += 1
        
        logger.info(f"Imported {added} questions, {duplicates} duplicates skipped")
        return added, duplicates
    
    def get_questions_by_topic(
        self,
        topic: str,
        year: Optional[int] = None,
        difficulty: Optional[str] = None,
        limit: int = 50
    ) -> List[StoredQuestion]:
        """Get all questions for a topic, optionally filtered."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM questions WHERE topic = ?"
            params = [topic]
            
            if year:
                query += " AND year = ?"
                params.append(year)
            
            if difficulty:
                query += " AND difficulty = ?"
                params.append(difficulty)
            
            query += " ORDER BY year DESC, id DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [StoredQuestion.from_row(row) for row in cursor.fetchall()]
    
    def get_similar_questions(
        self,
        topic: str,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 10
    ) -> List[StoredQuestion]:
        """
        Get similar questions for RAG retrieval.
        Used by question generator to find examples.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM questions WHERE topic = ?"
            params = [topic]
            
            if question_type and question_type != "unknown":
                query += " AND question_type = ?"
                params.append(question_type)
            
            if difficulty:
                query += " AND difficulty = ?"
                params.append(difficulty)
            
            # Prioritize high-confidence classifications
            query += " ORDER BY topic_confidence DESC, year DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [StoredQuestion.from_row(row) for row in cursor.fetchall()]
    
    def get_questions_by_section(
        self,
        section: str,
        limit: int = 100
    ) -> List[StoredQuestion]:
        """Get all questions in a section."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM questions WHERE section = ? ORDER BY year DESC LIMIT ?",
                (section, limit)
            )
            return [StoredQuestion.from_row(row) for row in cursor.fetchall()]
    
    def get_topic_history(self, topic: str) -> List[Dict]:
        """
        Get question count per year for a topic.
        Useful for trend analysis.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT year, COUNT(*) as count, 
                       AVG(CASE WHEN difficulty='easy' THEN 1 
                                WHEN difficulty='medium' THEN 2 
                                ELSE 3 END) as avg_difficulty
                FROM questions 
                WHERE topic = ? AND year IS NOT NULL
                GROUP BY year
                ORDER BY year
            ''', (topic,))
            
            return [
                {
                    'year': row['year'],
                    'count': row['count'],
                    'avg_difficulty': row['avg_difficulty']
                }
                for row in cursor.fetchall()
            ]
    
    def get_all_topics(self) -> List[Dict]:
        """Get all unique topics with counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT topic, section, COUNT(*) as count,
                       AVG(topic_confidence) as avg_confidence
                FROM questions
                GROUP BY topic, section
                ORDER BY count DESC
            ''')
            
            return [
                {
                    'topic': row['topic'],
                    'section': row['section'],
                    'count': row['count'],
                    'avg_confidence': row['avg_confidence']
                }
                for row in cursor.fetchall()
            ]
    
    def get_topic_questions_json(self, topic: str) -> str:
        """
        Get all questions for a topic as JSON.
        Used for HTML report dropdown feature.
        """
        questions = self.get_questions_by_topic(topic, limit=100)
        return json.dumps([q.to_dict() for q in questions], indent=2)
    
    def export_topic_questions_for_html(self, output_dir: Path):
        """
        Export questions per topic as JSON files.
        Creates one JSON file per topic for lazy loading in HTML.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        topics = self.get_all_topics()
        
        for topic_info in topics:
            topic = topic_info['topic']
            questions = self.get_questions_by_topic(topic, limit=100)
            
            # Prepare data for HTML display
            export_data = []
            for q in questions:
                export_data.append({
                    'year': q.year,
                    'shift': q.shift,
                    'text': q.question_text,
                    'options': q.options,
                    'difficulty': q.difficulty,
                    'question_type': q.question_type
                })
            
            file_path = output_dir / f"{topic}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(topics)} topic JSON files to {output_dir}")
        return len(topics)
    
    def get_statistics(self) -> Dict:
        """Get overall database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM questions")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT topic) FROM questions")
            topics = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT year) FROM questions WHERE year IS NOT NULL")
            years = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count 
                FROM questions 
                GROUP BY difficulty
            """)
            difficulty_dist = {row['difficulty']: row['count'] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT section, COUNT(*) as count 
                FROM questions 
                GROUP BY section
            """)
            section_dist = {row['section']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_questions': total,
                'unique_topics': topics,
                'years_covered': years,
                'difficulty_distribution': difficulty_dist,
                'section_distribution': section_dist
            }
    
    def clear_all(self):
        """Clear all questions (use with caution!)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM questions")
            cursor.execute("DELETE FROM topic_stats")
            logger.warning("All questions deleted from database")


def import_analysis_files(db: QuestionBankDB, analysis_dir: Path) -> Dict:
    """
    Import all analysis JSON files into the question bank.
    Returns import statistics.
    """
    analysis_dir = Path(analysis_dir)
    
    if not analysis_dir.exists():
        logger.warning(f"Analysis directory not found: {analysis_dir}")
        return {'files': 0, 'added': 0, 'duplicates': 0}
    
    stats = {'files': 0, 'added': 0, 'duplicates': 0}
    
    for json_file in analysis_dir.glob("*_analysis.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            added, dupes = db.add_from_analysis(analysis)
            stats['files'] += 1
            stats['added'] += added
            stats['duplicates'] += dupes
            
            logger.info(f"Imported {json_file.name}: {added} new, {dupes} duplicates")
            
        except Exception as e:
            logger.error(f"Error importing {json_file}: {e}")
    
    return stats


if __name__ == "__main__":
    # Test the question bank
    logging.basicConfig(level=logging.INFO)
    
    db = QuestionBankDB()
    
    # Add test question
    db.add_question(
        question_text="A train travels at 60 km/hr. How long will it take to cover 180 km?",
        options=["2 hours", "3 hours", "4 hours", "5 hours"],
        section="numerical_ability",
        topic="speed_time_distance",
        difficulty="easy",
        question_type="calculation",
        year=2023,
        shift=1,
        topic_confidence=0.85
    )
    
    # Get stats
    stats = db.get_statistics()
    print(f"\n📊 Question Bank Statistics:")
    print(f"   Total Questions: {stats['total_questions']}")
    print(f"   Unique Topics: {stats['unique_topics']}")
    print(f"   Years Covered: {stats['years_covered']}")
    
    # Query questions
    questions = db.get_questions_by_topic("speed_time_distance")
    print(f"\n📝 Speed/Time/Distance Questions: {len(questions)}")
    for q in questions[:3]:
        print(f"   - {q.question_text[:60]}...")
