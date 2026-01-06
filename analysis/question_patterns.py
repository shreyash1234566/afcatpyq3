"""
Question Pattern Analyzer
=========================
Analyzes patterns in historical AFCAT questions to:
1. Extract question templates and structures
2. Identify numerical ranges and common values
3. Detect distractor (wrong option) patterns
4. Find topic-specific language patterns
5. Build pattern library for question generation
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
import statistics

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.question_bank import QuestionBankDB, StoredQuestion

logger = logging.getLogger(__name__)


@dataclass
class QuestionTemplate:
    """A generalized question template."""
    template: str  # Question with placeholders like {number}, {unit}
    topic: str
    question_type: str
    difficulty: str
    placeholders: List[str]
    example_question: str
    frequency: int = 1


@dataclass
class NumberPattern:
    """Pattern for numbers used in questions."""
    topic: str
    placeholder_type: str  # speed, distance, time, money, etc.
    values: List[float]
    min_value: float
    max_value: float
    common_values: List[float]
    preferred_divisibility: List[int]  # Numbers divisible by these (e.g., 5, 10)


@dataclass
class DistractorPattern:
    """Pattern for generating wrong options."""
    topic: str
    pattern_type: str  # off_by_one, common_mistake, unit_error, etc.
    description: str
    examples: List[Dict]


@dataclass
class TopicPatternAnalysis:
    """Complete pattern analysis for a topic."""
    topic: str
    section: str
    total_questions: int
    templates: List[QuestionTemplate]
    number_patterns: List[NumberPattern]
    distractor_patterns: List[DistractorPattern]
    keywords: List[str]
    difficulty_distribution: Dict[str, float]
    question_type_distribution: Dict[str, float]
    average_length: float
    common_phrases: List[str]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['templates'] = [asdict(t) for t in self.templates]
        result['number_patterns'] = [asdict(n) for n in self.number_patterns]
        result['distractor_patterns'] = [asdict(d) for d in self.distractor_patterns]
        return result


class PatternExtractor:
    """
    Extracts patterns from questions for a specific topic.
    """
    
    # Regex patterns for extracting numbers with context
    NUMBER_PATTERNS = {
        'speed': r'(\d+\.?\d*)\s*(?:km/?hr?|m/?s|mph|kmph)',
        'distance': r'(\d+\.?\d*)\s*(?:km|m|meters?|miles?)',
        'time': r'(\d+\.?\d*)\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?)',
        'money': r'(?:Rs\.?|₹|INR)\s*(\d+\.?\d*)',
        'percentage': r'(\d+\.?\d*)\s*%',
        'ratio': r'(\d+)\s*:\s*(\d+)',
        'age': r'(\d+)\s*(?:years?|yrs?)\s*(?:old)?',
        'workers': r'(\d+)\s*(?:men|women|workers?|persons?|people)',
        'generic': r'\b(\d+\.?\d*)\b'
    }
    
    # Common question starters by type
    QUESTION_STARTERS = {
        'calculation': [
            r'^(?:find|calculate|what is|determine|compute)',
            r'^(?:how (?:much|many|long|far))',
        ],
        'comparison': [
            r'^(?:which|who|what) is (?:greater|smaller|more|less|larger)',
            r'^compare',
        ],
        'selection': [
            r'^(?:which of the following|choose|select)',
            r'^(?:identify|pick|point out)',
        ],
        'completion': [
            r'^(?:complete|fill in|supply)',
        ],
        'true_false': [
            r'^(?:which (?:statement|option) is (?:true|false|correct|incorrect))',
        ]
    }
    
    def __init__(self, question_bank: QuestionBankDB):
        self.db = question_bank
    
    def extract_template(self, question: str) -> Tuple[str, List[str]]:
        """
        Convert a question to a template by replacing specific values with placeholders.
        Returns (template, list of placeholders used).
        """
        template = question
        placeholders = []
        
        # Replace specific patterns with named placeholders
        replacements = [
            (self.NUMBER_PATTERNS['speed'], '{speed}'),
            (self.NUMBER_PATTERNS['distance'], '{distance}'),
            (self.NUMBER_PATTERNS['time'], '{time}'),
            (self.NUMBER_PATTERNS['money'], 'Rs. {amount}'),
            (self.NUMBER_PATTERNS['percentage'], '{percent}%'),
            (self.NUMBER_PATTERNS['age'], '{age} years'),
            (self.NUMBER_PATTERNS['workers'], '{workers}'),
        ]
        
        for pattern, placeholder in replacements:
            matches = re.findall(pattern, template, re.IGNORECASE)
            if matches:
                # Replace each match with numbered placeholder
                for i, match in enumerate(matches[:3]):  # Limit to 3 replacements
                    if isinstance(match, tuple):
                        match = match[0]
                    placeholder_name = placeholder.replace('{', '{' + str(i+1) + '_')
                    template = re.sub(
                        re.escape(str(match)) + r'(?=\s|$|[,.])',
                        placeholder_name.strip('{}'),
                        template,
                        count=1
                    )
                    placeholders.append(placeholder_name.strip('{}'))
        
        # Replace remaining generic numbers
        remaining_nums = re.findall(r'\b(\d+\.?\d*)\b', template)
        for i, num in enumerate(remaining_nums[:5]):
            template = template.replace(num, f'{{num{i+1}}}', 1)
            placeholders.append(f'num{i+1}')
        
        return template, placeholders
    
    def extract_numbers_with_context(self, question: str) -> Dict[str, List[float]]:
        """Extract numbers with their semantic context."""
        extracted = defaultdict(list)
        
        for num_type, pattern in self.NUMBER_PATTERNS.items():
            if num_type == 'generic':
                continue
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        try:
                            extracted[num_type].append(float(m))
                        except ValueError:
                            pass
                else:
                    try:
                        extracted[num_type].append(float(match))
                    except ValueError:
                        pass
        
        return dict(extracted)
    
    def detect_question_type(self, question: str) -> str:
        """Detect the type of question based on patterns."""
        question_lower = question.lower()
        
        for q_type, patterns in self.QUESTION_STARTERS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type
        
        # Default based on content
        if '?' in question:
            if re.search(r'how (?:much|many|long)', question_lower):
                return 'calculation'
        
        return 'unknown'
    
    def extract_keywords(self, questions: List[StoredQuestion]) -> List[str]:
        """Extract topic-specific keywords from questions."""
        # Combine all question texts
        all_text = ' '.join(q.question_text.lower() for q in questions)
        
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'if', 'then', 'than', 'that', 'this', 'these', 'those',
            'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
            'and', 'or', 'but', 'not', 'no', 'yes', 'all', 'each', 'every',
            'find', 'calculate', 'determine', 'given', 'following'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        word_counts = Counter(w for w in words if w not in stopwords)
        
        # Return most common keywords
        return [word for word, count in word_counts.most_common(30)]
    
    def analyze_distractors(self, questions: List[StoredQuestion]) -> List[DistractorPattern]:
        """Analyze patterns in wrong options."""
        patterns = []
        
        # Collect all options with their questions
        all_options = []
        for q in questions:
            if len(q.options) >= 4:
                # Try to identify which options are wrong
                # (without correct answer, we analyze option relationships)
                options = q.options
                all_options.append({
                    'question': q.question_text,
                    'options': options
                })
        
        # Analyze option relationships
        if len(all_options) >= 3:
            # Look for common distractor patterns
            patterns.append(DistractorPattern(
                topic=questions[0].topic if questions else 'unknown',
                pattern_type='numeric_neighbors',
                description='Options that are close to correct answer (+/- small value)',
                examples=[{'example': 'If answer is 100, distractors: 90, 110, 120'}]
            ))
            
            patterns.append(DistractorPattern(
                topic=questions[0].topic if questions else 'unknown',
                pattern_type='common_mistake',
                description='Values from common calculation errors',
                examples=[{'example': 'Forgetting to multiply by 2, using wrong formula'}]
            ))
        
        return patterns


class QuestionPatternAnalyzer:
    """
    Main analyzer that builds comprehensive pattern analysis for topics.
    """
    
    def __init__(self, question_bank: QuestionBankDB):
        self.db = question_bank
        self.extractor = PatternExtractor(question_bank)
    
    def analyze_topic(self, topic: str) -> Optional[TopicPatternAnalysis]:
        """
        Perform complete pattern analysis for a topic.
        """
        questions = self.db.get_questions_by_topic(topic, limit=100)
        
        if not questions:
            logger.warning(f"No questions found for topic: {topic}")
            return None
        
        logger.info(f"Analyzing {len(questions)} questions for topic: {topic}")
        
        # Extract templates
        templates = self._extract_templates(questions)
        
        # Analyze numbers
        number_patterns = self._analyze_numbers(questions, topic)
        
        # Analyze distractors
        distractor_patterns = self.extractor.analyze_distractors(questions)
        
        # Get keywords
        keywords = self.extractor.extract_keywords(questions)
        
        # Calculate distributions
        difficulty_dist = self._calculate_distribution(
            [q.difficulty for q in questions]
        )
        qtype_dist = self._calculate_distribution(
            [q.question_type for q in questions]
        )
        
        # Average question length
        avg_length = statistics.mean(len(q.question_text) for q in questions)
        
        # Common phrases
        common_phrases = self._extract_common_phrases(questions)
        
        # Get section from first question
        section = questions[0].section if questions else 'unknown'
        
        return TopicPatternAnalysis(
            topic=topic,
            section=section,
            total_questions=len(questions),
            templates=templates,
            number_patterns=number_patterns,
            distractor_patterns=distractor_patterns,
            keywords=keywords,
            difficulty_distribution=difficulty_dist,
            question_type_distribution=qtype_dist,
            average_length=avg_length,
            common_phrases=common_phrases
        )
    
    def _extract_templates(self, questions: List[StoredQuestion]) -> List[QuestionTemplate]:
        """Extract question templates from questions."""
        template_counts = Counter()
        template_examples = {}
        template_meta = {}
        
        for q in questions:
            template, placeholders = self.extractor.extract_template(q.question_text)
            
            # Normalize template for grouping
            normalized = re.sub(r'\d+', 'N', template)
            normalized = ' '.join(normalized.split())
            
            template_counts[normalized] += 1
            if normalized not in template_examples:
                template_examples[normalized] = q.question_text
                template_meta[normalized] = {
                    'placeholders': placeholders,
                    'topic': q.topic,
                    'question_type': q.question_type,
                    'difficulty': q.difficulty
                }
        
        # Return top templates
        templates = []
        for template, count in template_counts.most_common(10):
            if count >= 2 or len(template_counts) < 5:  # Include if frequency >= 2 or few templates
                meta = template_meta.get(template, {})
                templates.append(QuestionTemplate(
                    template=template,
                    topic=meta.get('topic', ''),
                    question_type=meta.get('question_type', 'unknown'),
                    difficulty=meta.get('difficulty', 'medium'),
                    placeholders=meta.get('placeholders', []),
                    example_question=template_examples.get(template, ''),
                    frequency=count
                ))
        
        return templates
    
    def _analyze_numbers(self, questions: List[StoredQuestion], topic: str) -> List[NumberPattern]:
        """Analyze numerical patterns in questions."""
        number_data = defaultdict(list)
        
        for q in questions:
            extracted = self.extractor.extract_numbers_with_context(q.question_text)
            for num_type, values in extracted.items():
                number_data[num_type].extend(values)
        
        patterns = []
        for num_type, values in number_data.items():
            if not values:
                continue
            
            # Calculate statistics
            unique_values = list(set(values))
            common_values = [v for v, c in Counter(values).most_common(5)]
            
            # Find preferred divisibility
            divisibility = []
            for d in [5, 10, 2, 3]:
                divisible_count = sum(1 for v in values if v % d == 0)
                if divisible_count > len(values) * 0.5:
                    divisibility.append(d)
            
            patterns.append(NumberPattern(
                topic=topic,
                placeholder_type=num_type,
                values=unique_values[:20],
                min_value=min(values) if values else 0,
                max_value=max(values) if values else 100,
                common_values=common_values,
                preferred_divisibility=divisibility
            ))
        
        return patterns
    
    def _calculate_distribution(self, items: List[str]) -> Dict[str, float]:
        """Calculate percentage distribution of items."""
        if not items:
            return {}
        
        counts = Counter(items)
        total = len(items)
        return {k: round(v / total * 100, 1) for k, v in counts.items()}
    
    def _extract_common_phrases(self, questions: List[StoredQuestion]) -> List[str]:
        """Extract commonly used phrases (2-4 word combinations)."""
        all_text = ' '.join(q.question_text.lower() for q in questions)
        
        # Extract 2-gram and 3-gram phrases
        words = all_text.split()
        phrases = []
        
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if len(phrase) > 5 and not phrase.isdigit():
                    phrases.append(phrase)
        
        # Return most common
        phrase_counts = Counter(phrases)
        return [p for p, c in phrase_counts.most_common(15) if c >= 2]
    
    def analyze_all_topics(self) -> Dict[str, TopicPatternAnalysis]:
        """Analyze patterns for all topics in the database."""
        topics = self.db.get_all_topics()
        results = {}
        
        for topic_info in topics:
            topic = topic_info['topic']
            if topic and topic != 'unknown':
                analysis = self.analyze_topic(topic)
                if analysis:
                    results[topic] = analysis
        
        return results
    
    def export_patterns(self, output_path: Path):
        """Export all pattern analysis to JSON."""
        analyses = self.analyze_all_topics()
        
        export_data = {
            topic: analysis.to_dict()
            for topic, analysis in analyses.items()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported patterns for {len(analyses)} topics to {output_path}")
        return len(analyses)


def generate_pattern_report(analysis: TopicPatternAnalysis) -> str:
    """Generate a human-readable pattern report for a topic."""
    report = []
    report.append(f"=" * 60)
    report.append(f"PATTERN ANALYSIS: {analysis.topic.replace('_', ' ').upper()}")
    report.append(f"=" * 60)
    report.append(f"\nSection: {analysis.section}")
    report.append(f"Total Questions Analyzed: {analysis.total_questions}")
    report.append(f"Average Question Length: {analysis.average_length:.0f} chars")
    
    report.append(f"\n📊 DIFFICULTY DISTRIBUTION:")
    for diff, pct in analysis.difficulty_distribution.items():
        bar = "█" * int(pct / 5)
        report.append(f"  {diff:10s}: {bar} {pct:.1f}%")
    
    report.append(f"\n📝 QUESTION TYPES:")
    for qtype, pct in analysis.question_type_distribution.items():
        report.append(f"  {qtype}: {pct:.1f}%")
    
    if analysis.templates:
        report.append(f"\n🔧 TOP QUESTION TEMPLATES:")
        for i, t in enumerate(analysis.templates[:5], 1):
            report.append(f"\n  Template {i} (used {t.frequency}x):")
            report.append(f"  {t.template[:100]}...")
    
    if analysis.number_patterns:
        report.append(f"\n🔢 NUMBER PATTERNS:")
        for np in analysis.number_patterns[:5]:
            report.append(f"  {np.placeholder_type}:")
            report.append(f"    Range: {np.min_value} - {np.max_value}")
            report.append(f"    Common: {np.common_values[:5]}")
    
    report.append(f"\n🔑 KEYWORDS:")
    report.append(f"  {', '.join(analysis.keywords[:15])}")
    
    if analysis.common_phrases:
        report.append(f"\n💬 COMMON PHRASES:")
        for phrase in analysis.common_phrases[:10]:
            report.append(f"  - \"{phrase}\"")
    
    return '\n'.join(report)


if __name__ == "__main__":
    # Test the pattern analyzer
    logging.basicConfig(level=logging.INFO)
    
    db = QuestionBankDB()
    analyzer = QuestionPatternAnalyzer(db)
    
    # Analyze a topic
    print("\n🔍 Analyzing question patterns...")
    
    analysis = analyzer.analyze_topic("speed_time_distance")
    
    if analysis:
        report = generate_pattern_report(analysis)
        print(report)
    else:
        print("No questions found for analysis. Add questions to the database first.")
    
    # Get all topics
    topics = db.get_all_topics()
    print(f"\n📚 Topics in database: {len(topics)}")
    for t in topics[:10]:
        print(f"   {t['topic']}: {t['count']} questions")
