"""
AI Question Generator with Ollama
==================================
Generates predicted AFCAT exam questions using:
1. RAG (Retrieval Augmented Generation) - retrieves similar past questions
2. Pattern Analysis - identifies question templates and structures
3. Ollama LLM - generates new questions based on patterns and difficulty

Supports local Ollama models (llama3, mistral, etc.) for free, offline generation.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import requests

# Import from project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.question_bank import QuestionBankDB, StoredQuestion

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """A generated predicted question."""
    question_text: str
    options: List[str]
    correct_answer: Optional[str]
    section: str
    topic: str
    predicted_difficulty: str
    question_type: str
    generation_reason: str
    similar_past_questions: List[str]
    confidence: float
    pattern_used: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of question generation for a topic."""
    topic: str
    section: str
    target_difficulty: str
    target_count: int
    questions: List[GeneratedQuestion]
    generation_stats: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['questions'] = [q.to_dict() for q in self.questions]
        return result


class OllamaClient:
    """
    Client for Ollama API.
    Ollama must be running locally (ollama serve).
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 120
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        if self._available is not None:
            return self._available
            
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = response.status_code == 200
            if self._available:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                logger.info(f"Ollama available with models: {model_names}")
                if self.model not in model_names and f"{self.model}:latest" not in [m.get('name', '') for m in models]:
                    logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
        
        return self._available
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate text using Ollama.
        Returns None if generation fails.
        """
        if not self.is_available():
            return None
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m.get('name', '') for m in models]
        except:
            pass
        return []


class QuestionPatternExtractor:
    """
    Extracts patterns and templates from past questions.
    Used to guide question generation.
    """
    
    # Common question patterns by topic
    PATTERN_TEMPLATES = {
        "speed_time_distance": [
            "A {vehicle} travels at {speed} km/hr. How long will it take to cover {distance} km?",
            "A {vehicle} covers {distance} km in {time} hours. What is its speed?",
            "Two trains start from stations A and B towards each other at {speed1} and {speed2} km/hr. If the distance is {distance} km, when will they meet?",
            "A boat goes {distance1} km upstream in {time1} hours and {distance2} km downstream in {time2} hours. Find the speed of the stream.",
        ],
        "time_and_work": [
            "A can complete a work in {days_a} days and B can complete it in {days_b} days. In how many days can they complete it together?",
            "If {workers} men can do a work in {days} days, how many men are required to complete the work in {target_days} days?",
            "{workers_a} men and {workers_b} women can complete a work in {days} days. If {ratio} men equal {ratio_w} women in work capacity, find the time taken by 1 man alone.",
        ],
        "profit_loss": [
            "An article is bought for Rs. {cp} and sold for Rs. {sp}. Find the profit/loss percentage.",
            "A shopkeeper marks an article {markup}% above cost price and gives {discount}% discount. Find the profit percentage.",
            "By selling an article for Rs. {sp}, a trader gains {profit}%. What was the cost price?",
        ],
        "percentages": [
            "What is {percent}% of {number}?",
            "If {number1} is {percent}% more than {number2}, find {number1}.",
            "The price of an article increased by {percent1}% and then decreased by {percent2}%. Find the net change.",
        ],
        "ratio_proportion": [
            "Divide Rs. {total} among A, B, C in the ratio {ratio}.",
            "The ratio of A to B is {ratio1} and B to C is {ratio2}. Find A:B:C.",
            "If {quantity} is divided in the ratio {ratio}, find the difference between the parts.",
        ],
        "simple_compound_interest": [
            "Find the simple interest on Rs. {principal} at {rate}% per annum for {time} years.",
            "Find the compound interest on Rs. {principal} at {rate}% per annum for {time} years.",
            "In how many years will Rs. {principal} become Rs. {amount} at {rate}% simple interest?",
        ],
    }
    
    def __init__(self, question_bank: QuestionBankDB):
        self.db = question_bank
    
    def extract_patterns(self, topic: str, limit: int = 20) -> Dict:
        """
        Extract patterns from past questions on a topic.
        Returns pattern analysis for question generation.
        """
        questions = self.db.get_questions_by_topic(topic, limit=limit)
        
        if not questions:
            return {
                'topic': topic,
                'patterns': self.PATTERN_TEMPLATES.get(topic, []),
                'difficulty_distribution': {'easy': 33, 'medium': 34, 'hard': 33},
                'common_numbers': [],
                'question_types': ['calculation'],
                'sample_questions': []
            }
        
        # Analyze difficulty distribution
        difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        for q in questions:
            difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
        
        total = sum(difficulty_counts.values())
        difficulty_dist = {k: int(v/total*100) if total > 0 else 33 
                          for k, v in difficulty_counts.items()}
        
        # Extract numbers commonly used
        numbers = []
        for q in questions:
            nums = re.findall(r'\b\d+\.?\d*\b', q.question_text)
            numbers.extend([float(n) for n in nums if float(n) < 10000])
        
        # Get unique question types
        q_types = list(set(q.question_type for q in questions if q.question_type))
        
        # Sample questions for few-shot prompting
        samples = [
            {
                'text': q.question_text,
                'options': q.options,
                'difficulty': q.difficulty
            }
            for q in questions[:5]
        ]
        
        return {
            'topic': topic,
            'patterns': self.PATTERN_TEMPLATES.get(topic, []),
            'difficulty_distribution': difficulty_dist,
            'common_numbers': sorted(set(numbers))[:20] if numbers else [10, 20, 50, 100],
            'question_types': q_types if q_types else ['calculation'],
            'sample_questions': samples
        }
    
    def get_similar_questions(
        self,
        topic: str,
        difficulty: str,
        count: int = 5
    ) -> List[StoredQuestion]:
        """Get similar questions for RAG context."""
        return self.db.get_similar_questions(
            topic=topic,
            difficulty=difficulty,
            limit=count
        )


class AFCATQuestionGenerator:
    """
    Main question generator using Ollama + RAG.
    
    Workflow:
    1. Retrieve similar past questions (RAG)
    2. Analyze patterns and difficulty
    3. Generate new questions using Ollama
    4. Parse and validate generated questions
    """
    
    SYSTEM_PROMPT = """You are an expert AFCAT (Air Force Common Admission Test) exam question generator.
Your task is to generate realistic multiple-choice questions for the Indian Air Force entrance exam.

IMPORTANT GUIDELINES:
1. Questions must be exam-appropriate - clear, unambiguous, and solvable
2. All 4 options must be plausible (no obviously wrong answers)
3. Match the specified difficulty level:
   - EASY: Direct formula application, simple calculations
   - MEDIUM: 2-3 step problems, moderate complexity
   - HARD: Multi-step reasoning, complex scenarios
4. Use realistic Indian context (Rs., km, Indian places, etc.)
5. Numbers should be reasonable and computationally manageable
6. Each question must have exactly ONE correct answer

OUTPUT FORMAT:
Return questions in this exact JSON format:
{
    "questions": [
        {
            "question": "Full question text here?",
            "options": ["(A) option1", "(B) option2", "(C) option3", "(D) option4"],
            "correct": "A",
            "explanation": "Brief solution explanation"
        }
    ]
}"""
    
    TOPIC_SECTIONS = {
        "speed_time_distance": "numerical_ability",
        "time_and_work": "numerical_ability",
        "profit_loss": "numerical_ability",
        "percentages": "numerical_ability",
        "ratio_proportion": "numerical_ability",
        "simple_compound_interest": "numerical_ability",
        "averages": "numerical_ability",
        "number_series": "numerical_ability",
        "algebra": "numerical_ability",
        "geometry": "numerical_ability",
        "mensuration": "numerical_ability",
        "probability": "numerical_ability",
        "data_interpretation": "numerical_ability",
        "synonyms": "english",
        "antonyms": "english",
        "sentence_completion": "english",
        "reading_comprehension": "english",
        "error_detection": "english",
        "idioms_phrases": "english",
        "analogies": "reasoning",
        "syllogisms": "reasoning",
        "blood_relations": "reasoning",
        "coding_decoding": "reasoning",
        "direction_sense": "reasoning",
        "series_completion": "reasoning",
        "indian_history": "general_awareness",
        "indian_geography": "general_awareness",
        "indian_polity": "general_awareness",
        "indian_economy": "general_awareness",
        "physics": "general_awareness",
        "chemistry": "general_awareness",
        "biology": "general_awareness",
        "current_affairs": "general_awareness",
        "defence_awareness": "general_awareness",
    }
    
    def __init__(
        self,
        question_bank: QuestionBankDB,
        ollama_model: str = "llama3",
        ollama_url: str = "http://localhost:11434"
    ):
        self.db = question_bank
        self.pattern_extractor = QuestionPatternExtractor(question_bank)
        self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model)
        # Semantic retrieval
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.semantic_model = SentenceTransformer("all-mpnet-base-v2")
            self.faiss = faiss
            self.semantic_index = None
            self.semantic_questions = []
            logger.info("Semantic retrieval enabled (all-mpnet-base-v2)")
        except Exception as e:
            self.semantic_model = None
            self.faiss = None
            self.semantic_index = None
            self.semantic_questions = []
            logger.warning(f"Semantic retrieval disabled: {e}")
        logger.info(f"Question Generator initialized with model: {ollama_model}")

    def build_semantic_index(self, questions: list):
        if not self.semantic_model or not self.faiss:
            return
        texts = [q['question_text'] for q in questions if 'question_text' in q]
        embeddings = self.semantic_model.encode(texts, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = self.faiss.IndexFlatIP(dim)
        self.faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        self.semantic_index = index
        self.semantic_questions = questions

    def semantic_search(self, query: str, top_k: int = 5, topic: str = None):
        if not self.semantic_model or not self.semantic_index:
            return []
        query_emb = self.semantic_model.encode([query])
        self.faiss.normalize_L2(query_emb)
        scores, idxs = self.semantic_index.search(query_emb.astype('float32'), top_k * 2)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.semantic_questions):
                q = self.semantic_questions[idx]
                if topic and topic.lower() not in q.get('topic', '').lower():
                    continue
                results.append((q, float(score)))
                if len(results) >= top_k:
                    break
        return results
    
    def _build_generation_prompt(
        self,
        topic: str,
        difficulty: str,
        count: int,
        patterns: Dict,
        similar_questions: List[StoredQuestion]
    ) -> str:
        """Build the prompt for question generation."""
        
        topic_display = topic.replace('_', ' ').title()
        
        # Format similar questions as examples
        examples = ""
        if similar_questions:
            examples = "\n\nHere are examples of real AFCAT questions on this topic:\n"
            for i, q in enumerate(similar_questions[:5], 1):
                examples += f"\nExample {i} ({q.difficulty}):\n{q.question_text}\n"
                for opt in q.options:
                    examples += f"  {opt}\n"
        
        # Format patterns
        pattern_hints = ""
        if patterns.get('patterns'):
            pattern_hints = "\n\nCommon question patterns for this topic:\n"
            for p in patterns['patterns'][:3]:
                pattern_hints += f"- {p}\n"
        
        prompt = f"""Generate {count} NEW and UNIQUE multiple-choice questions for AFCAT exam.

TOPIC: {topic_display}
DIFFICULTY: {difficulty.upper()}
SECTION: {self.TOPIC_SECTIONS.get(topic, 'general')}

{examples}
{pattern_hints}

REQUIREMENTS:
- Generate exactly {count} questions
- Difficulty must be {difficulty.upper()}
- Questions must be different from the examples
- Use Indian context (Rs., km, Indian cities, etc.)
- Each question must have 4 options labeled (A), (B), (C), (D)
- Include the correct answer and brief explanation

Return ONLY valid JSON in the specified format."""

        return prompt
    
    def _parse_generated_questions(
        self,
        response: str,
        topic: str,
        difficulty: str
    ) -> List[GeneratedQuestion]:
        """Parse LLM response into GeneratedQuestion objects."""
        questions = []
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning("No JSON found in response")
                return questions
            
            data = json.loads(json_match.group())
            
            for q in data.get('questions', []):
                # Parse options
                options = q.get('options', [])
                if isinstance(options, list):
                    # Clean options (remove (A), (B) prefixes if present)
                    clean_options = []
                    for opt in options:
                        clean = re.sub(r'^\([A-Da-d]\)\s*', '', str(opt).strip())
                        clean_options.append(clean)
                    options = clean_options
                
                # Get correct answer
                correct = q.get('correct', 'A')
                if correct in ['A', 'B', 'C', 'D'] and len(options) >= 4:
                    correct_idx = ord(correct.upper()) - ord('A')
                    correct_answer = options[correct_idx] if correct_idx < len(options) else None
                else:
                    correct_answer = None
                
                gen_q = GeneratedQuestion(
                    question_text=q.get('question', ''),
                    options=options,
                    correct_answer=correct_answer,
                    section=self.TOPIC_SECTIONS.get(topic, 'general'),
                    topic=topic,
                    predicted_difficulty=difficulty,
                    question_type='generated',
                    generation_reason=f"Generated for {topic} at {difficulty} difficulty",
                    similar_past_questions=[],
                    confidence=0.75,
                    pattern_used=None
                )
                
                # Validate question
                if len(gen_q.question_text) > 10 and len(gen_q.options) >= 4:
                    questions.append(gen_q)
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response was: {response[:500]}")
        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
        
        return questions
    
    def generate_questions(
        self,
        topic: str,
        difficulty: str = "medium",
        count: int = 3,
        use_rag: bool = True
    ) -> GenerationResult:
        """
        Generate predicted questions for a topic.
        
        Args:
            topic: The topic to generate questions for
            difficulty: easy, medium, or hard
            count: Number of questions to generate
            use_rag: Whether to use similar past questions as context
        
        Returns:
            GenerationResult with generated questions
        """
        logger.info(f"Generating {count} {difficulty} questions for {topic}")
        
        # Get patterns and similar questions
        patterns = self.pattern_extractor.extract_patterns(topic)
        similar = []
        if use_rag:
            similar = self.pattern_extractor.get_similar_questions(topic, difficulty, count=5)
        
        # Check Ollama availability
        if not self.ollama.is_available():
            logger.warning("Ollama not available, using fallback generation")
            return self._fallback_generation(topic, difficulty, count, patterns)
        
        # Build prompt and generate
        prompt = self._build_generation_prompt(topic, difficulty, count, patterns, similar)
        
        response = self.ollama.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7
        )
        
        if not response:
            logger.warning("Ollama returned empty response, using fallback")
            return self._fallback_generation(topic, difficulty, count, patterns)
        
        # Parse questions
        questions = self._parse_generated_questions(response, topic, difficulty)
        
        # Add similar question references
        for q in questions:
            q.similar_past_questions = [sq.question_text[:80] + "..." 
                                        for sq in similar[:3]]
        
        return GenerationResult(
            topic=topic,
            section=self.TOPIC_SECTIONS.get(topic, 'general'),
            target_difficulty=difficulty,
            target_count=count,
            questions=questions,
            generation_stats={
                'requested': count,
                'generated': len(questions),
                'rag_examples_used': len(similar),
                'model': self.ollama.model,
                'success': len(questions) >= count * 0.5
            }
        )
    
    def _fallback_generation(
        self,
        topic: str,
        difficulty: str,
        count: int,
        patterns: Dict
    ) -> GenerationResult:
        """
        Fallback generation when Ollama is not available.
        Uses pattern templates with random values.
        """
        import random
        
        questions = []
        templates = patterns.get('patterns', [])
        
        if not templates:
            # Generic fallback
            templates = [
                f"This is a sample {topic.replace('_', ' ')} question. What is the answer?",
            ]
        
        for i in range(min(count, len(templates))):
            template = templates[i % len(templates)]
            
            # Simple template filling (basic fallback)
            filled = template
            for match in re.finditer(r'\{(\w+)\}', template):
                placeholder = match.group(1)
                # Generate reasonable random values
                if 'speed' in placeholder or 'rate' in placeholder:
                    value = random.choice([20, 30, 40, 50, 60, 80])
                elif 'distance' in placeholder or 'time' in placeholder:
                    value = random.choice([100, 150, 200, 300, 500])
                elif 'days' in placeholder:
                    value = random.choice([5, 10, 12, 15, 20])
                elif 'percent' in placeholder:
                    value = random.choice([5, 10, 15, 20, 25])
                else:
                    value = random.choice([10, 20, 50, 100])
                filled = filled.replace(match.group(0), str(value))
            
            q = GeneratedQuestion(
                question_text=filled,
                options=["Option A", "Option B", "Option C", "Option D"],
                correct_answer="Option A",
                section=self.TOPIC_SECTIONS.get(topic, 'general'),
                topic=topic,
                predicted_difficulty=difficulty,
                question_type='template',
                generation_reason="Fallback template generation (Ollama unavailable)",
                similar_past_questions=[],
                confidence=0.3,
                pattern_used=template
            )
            questions.append(q)
        
        return GenerationResult(
            topic=topic,
            section=self.TOPIC_SECTIONS.get(topic, 'general'),
            target_difficulty=difficulty,
            target_count=count,
            questions=questions,
            generation_stats={
                'requested': count,
                'generated': len(questions),
                'rag_examples_used': 0,
                'model': 'fallback_template',
                'success': False,
                'warning': 'Ollama not available, used template fallback'
            }
        )
    
    def generate_for_predictions(
        self,
        predictions: Dict[str, List[Dict]],
        questions_per_topic: int = 2
    ) -> Dict[str, List[GenerationResult]]:
        """
        Generate questions for all predicted high-priority topics.
        
        Args:
            predictions: Dict of section -> list of topic predictions
            questions_per_topic: How many questions to generate per topic
        
        Returns:
            Dict of section -> list of GenerationResults
        """
        results = {}
        
        for section, topic_predictions in predictions.items():
            results[section] = []
            
            # Take top 5 topics per section
            for tp in topic_predictions[:5]:
                topic = tp.get('topic', tp.get('name', ''))
                confidence = tp.get('confidence', 0.5)
                
                # Determine difficulty based on prediction
                if confidence > 0.7:
                    difficulty = 'medium'  # High confidence = likely medium difficulty
                elif confidence > 0.5:
                    difficulty = 'hard'    # Medium confidence = might be harder
                else:
                    difficulty = 'easy'    # Lower priority topics tend to be easier
                
                try:
                    result = self.generate_questions(
                        topic=topic,
                        difficulty=difficulty,
                        count=questions_per_topic
                    )
                    results[section].append(result)
                    logger.info(f"Generated {len(result.questions)} questions for {topic}")
                    
                except Exception as e:
                    logger.error(f"Error generating for {topic}: {e}")
        
        return results


def export_generated_questions(
    results: Dict[str, List[GenerationResult]],
    output_path: Path,
    format: str = "json"
) -> Path:
    """Export generated questions to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        export_data = {}
        for section, gen_results in results.items():
            export_data[section] = [r.to_dict() for r in gen_results]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    elif format == "html":
        html = generate_questions_html(results)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    logger.info(f"Exported generated questions to {output_path}")
    return output_path


def generate_questions_html(results: Dict[str, List[GenerationResult]]) -> str:
    """Generate HTML report for generated questions."""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AFCAT Predicted Questions</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #1a365d; text-align: center; }
        h2 { color: #2d3748; border-bottom: 2px solid #4299e1; padding-bottom: 10px; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .question { background: #f7fafc; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #4299e1; }
        .question-text { font-weight: bold; margin-bottom: 10px; }
        .options { margin-left: 20px; }
        .option { margin: 5px 0; padding: 5px 10px; background: white; border-radius: 4px; }
        .option.correct { background: #c6f6d5; }
        .meta { font-size: 0.9em; color: #718096; margin-top: 10px; }
        .difficulty-easy { color: #38a169; }
        .difficulty-medium { color: #d69e2e; }
        .difficulty-hard { color: #e53e3e; }
        .topic-tag { display: inline-block; padding: 2px 8px; background: #bee3f8; border-radius: 4px; font-size: 0.85em; }
        .warning { background: #fef3c7; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>🎯 AFCAT Predicted Questions</h1>
    <div class="warning">
        ⚠️ These questions are AI-generated predictions based on historical patterns. 
        Use for practice, not as guaranteed exam content.
    </div>
"""
    
    for section, gen_results in results.items():
        html += f'<div class="section">\n'
        html += f'<h2>📚 {section.replace("_", " ").title()}</h2>\n'
        
        for result in gen_results:
            html += f'<h3><span class="topic-tag">{result.topic.replace("_", " ").title()}</span></h3>\n'
            
            for i, q in enumerate(result.questions, 1):
                diff_class = f"difficulty-{q.predicted_difficulty}"
                html += f'''
                <div class="question">
                    <div class="question-text">Q{i}. {q.question_text}</div>
                    <div class="options">
'''
                for j, opt in enumerate(q.options):
                    letter = chr(65 + j)
                    is_correct = q.correct_answer and opt == q.correct_answer
                    correct_class = " correct" if is_correct else ""
                    html += f'                        <div class="option{correct_class}">({letter}) {opt}</div>\n'
                
                html += f'''                    </div>
                    <div class="meta">
                        Difficulty: <span class="{diff_class}">{q.predicted_difficulty.title()}</span> | 
                        Confidence: {q.confidence:.0%}
                    </div>
                </div>
'''
        
        html += '</div>\n'
    
    html += """
    <footer style="text-align: center; margin-top: 40px; color: #718096;">
        <p>Generated using AI pattern analysis of previous AFCAT exams.</p>
    </footer>
</body>
</html>"""
    
    return html


if __name__ == "__main__":
    # Test the question generator
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    db = QuestionBankDB()
    generator = AFCATQuestionGenerator(db, ollama_model="llama3")
    
    # Check Ollama
    print(f"\n🔍 Ollama available: {generator.ollama.is_available()}")
    if generator.ollama.is_available():
        print(f"   Models: {generator.ollama.list_models()}")
    
    # Generate sample questions
    print("\n📝 Generating sample questions...")
    result = generator.generate_questions(
        topic="speed_time_distance",
        difficulty="medium",
        count=2
    )
    
    print(f"\n✅ Generated {len(result.questions)} questions:")
    for q in result.questions:
        print(f"\n   Q: {q.question_text}")
        for i, opt in enumerate(q.options):
            print(f"      ({chr(65+i)}) {opt}")
        print(f"   Difficulty: {q.predicted_difficulty}")
