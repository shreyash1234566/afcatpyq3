"""
Complete Exam Paper Analysis Pipeline
======================================
End-to-end pipeline for:
1. PDF/Image → OCR Text Extraction
2. Question Parsing & Structure Extraction
3. Topic Classification
4. Difficulty Prediction
5. Export to structured JSON for prediction model
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

# Import pipeline components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ocr_engine import ExamPaperOCR, MCQExtractor, OCREngine, ExtractedQuestion
from models.question_classifier import AFCATTopicClassifier, QuestionTypeClassifier, MathFormulaHandler
from models.enhanced_difficulty import EnhancedDifficultyPredictor

# Import preprocessing utilities
try:
    from utils.ocr_preprocessing import (
        diagnose_extraction,
        print_diagnostic_report,
        compute_question_quality,
        is_placeholder_question
    )
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalyzedQuestion:
    """Complete analysis of a single question."""
    question_number: int
    text: str
    options: List[str]
    section: str
    topic: str
    topic_label: str
    subtopic: Optional[str]
    question_type: str
    difficulty: str
    confidence_scores: Dict[str, float]
    has_formula: bool
    has_diagram_reference: bool
    image_refs: List[Dict]
    source_file: str
    confidence: float = 0.0  # Extraction confidence
    page_number: int = 0
    year: Optional[int] = None
    shift: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass  
class PaperAnalysisResult:
    """Complete analysis result of an exam paper."""
    source_file: str
    year: Optional[int]
    shift: Optional[int]
    extraction_date: str
    total_questions: int
    questions: List[AnalyzedQuestion]
    section_breakdown: Dict[str, int]
    topic_breakdown: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    ocr_confidence: float
    extraction_method: str
    page_images: Dict[int, List[Dict]]
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with nested question dicts."""
        result = asdict(self)
        result['questions'] = [q.to_dict() if hasattr(q, 'to_dict') else q 
                               for q in self.questions]
        return result


class ReviewPriority(Enum):
    """Priority levels for manual review."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class QualityCheck:
    """Quality check result for a question."""
    needs_review: bool
    priority: ReviewPriority
    issues: List[str]
    suggestions: List[str]


class QualityChecker:
    """Flags questions that need manual review."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_question_length: int = 15,
        min_options: int = 2
    ):
        self.confidence_threshold = confidence_threshold
        self.min_question_length = min_question_length
        self.min_options = min_options
        
    def check_question(self, analyzed: AnalyzedQuestion) -> QualityCheck:
        """Check if question needs manual review."""
        issues = []
        suggestions = []
        
        # Low topic confidence
        topic_conf = analyzed.confidence_scores.get("topic", 0)
        if topic_conf < self.confidence_threshold:
            issues.append(f"low_topic_confidence ({topic_conf:.2f})")
            suggestions.append("Verify topic classification manually")
            
        # Low difficulty confidence
        diff_conf = analyzed.confidence_scores.get("difficulty", 0)
        if diff_conf < self.confidence_threshold:
            issues.append(f"low_difficulty_confidence ({diff_conf:.2f})")
            
        # Unknown classifications
        if analyzed.topic == "unknown":
            issues.append("unknown_topic")
            suggestions.append("Assign topic manually")
            
        # Very short questions (likely extraction error)
        if len(analyzed.text) < self.min_question_length:
            issues.append("question_too_short")
            suggestions.append("Check if question was extracted correctly")
            
        # No options extracted
        if len(analyzed.options) < self.min_options:
            issues.append("insufficient_options")
            suggestions.append("Extract options manually from source")
            
        # Formula detection (verify extraction)
        if analyzed.has_formula:
            issues.append("contains_formula")
            suggestions.append("Verify mathematical notation is correct")
            
        # Diagram reference (content may be incomplete)
        if analyzed.has_diagram_reference:
            issues.append("references_diagram")
            suggestions.append("Question may require visual content")
            
        # Determine priority
        if len(issues) >= 3:
            priority = ReviewPriority.HIGH
        elif len(issues) >= 2:
            priority = ReviewPriority.MEDIUM
        elif len(issues) >= 1:
            priority = ReviewPriority.LOW
        else:
            priority = ReviewPriority.NONE
            
        return QualityCheck(
            needs_review=len(issues) > 0,
            priority=priority,
            issues=issues,
            suggestions=suggestions
        )


class AFCATExamAnalyzer:
    """
    End-to-end pipeline for analyzing AFCAT exam papers.
    Extracts questions from PDF/images and classifies them.
    """
    
    def __init__(
        self,
        ocr_engine: OCREngine = OCREngine.EASYOCR,
        use_transformers: bool = False,
        gpu: bool = False,
        output_dir: Path = None,
        quality_level: str = "standard"  # "fast", "standard", "max_quality"
    ):
        """
        Initialize the analyzer.
        
        Args:
            ocr_engine: Which OCR engine to use
            use_transformers: Whether to use transformer models for classification
            gpu: Whether to use GPU acceleration
            output_dir: Directory for saving results
            quality_level: OCR quality level - "fast", "standard", or "max_quality"
        """
        self.ocr = ExamPaperOCR(engine=ocr_engine, gpu=gpu, quality_level=quality_level)
        self.mcq_extractor = MCQExtractor()
        self.topic_classifier = AFCATTopicClassifier(use_transformers=use_transformers)
        self.type_classifier = QuestionTypeClassifier()
        self.difficulty_predictor = EnhancedDifficultyPredictor()
        self.math_handler = MathFormulaHandler()
        self.quality_checker = QualityChecker()
        
        self.output_dir = output_dir or Path("output/extracted")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized AFCATExamAnalyzer with {ocr_engine.value} engine, quality={quality_level}")
        
    def analyze_paper(
        self,
        file_path: Union[str, Path],
        year: Optional[int] = None,
        shift: Optional[int] = None,
        zone_mode: Optional[str] = None,
        single_section: Optional[str] = None,
        save_result: bool = True
    ) -> PaperAnalysisResult:
        """
        Analyze a complete exam paper.
        
        Args:
            file_path: Path to PDF or image file
            year: Exam year (for metadata)
            shift: Exam shift number (for metadata)
            save_result: Whether to save JSON result
            
        Returns:
            Complete analysis result
        """
        file_path = Path(file_path)
        logger.info(f"📄 Analyzing paper: {file_path.name}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        warnings = []
        
        # Step 1: Extract text via OCR
        logger.info("Step 1/4: Extracting text (OCR)...")
        
        if file_path.suffix.lower() == '.pdf':
            ocr_results = self.ocr.extract_from_pdf(file_path)
            extraction_method = "pdf_ocr"
            page_images = self.ocr.get_page_images()
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            ocr_results = self.ocr.extract_from_image(file_path)
            extraction_method = "image_ocr"
            page_images = {}
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        if not ocr_results:
            raise ValueError("No text extracted from document")
            
        # Calculate OCR confidence
        confidences = [r.confidence for r in ocr_results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        if avg_confidence < 0.7:
            warnings.append(f"Low OCR confidence: {avg_confidence:.1%}")
            
        logger.info(f"   Extracted {len(ocr_results)} text blocks (confidence: {avg_confidence:.1%})")
        
        # Step 2: Extract questions from text
        logger.info("Step 2/4: Parsing questions...")
        
        full_text = self.ocr.get_full_text(ocr_results)
        extracted_questions = self.mcq_extractor.extract_questions(full_text)
        
        if not extracted_questions:
            warnings.append("No questions could be extracted automatically")
            logger.warning("   ⚠️ No questions extracted!")
        else:
            logger.info(f"   Found {len(extracted_questions)} questions")
        
        # Update classifier with total question count for zone enforcement
        # Use the highest question number to handle off-by-one extraction cases
        total_questions = max((q.question_number for q in extracted_questions), default=0)
        self.topic_classifier.total_questions = total_questions
        if zone_mode:
            self.topic_classifier.zone_mode = zone_mode
        # Apply optional single-section override for short papers (~30Q)
        if single_section:
            self.topic_classifier.single_section_override = single_section
        else:
            self.topic_classifier.single_section_override = None
        if total_questions >= 80:
            logger.info(f"   Zone enforcement: {self.topic_classifier.zone_mode.upper()} (full paper)")
        elif total_questions <= 35:
            if single_section:
                logger.info(f"   Zone enforcement: OFF (single-section override: {single_section})")
            else:
                logger.info("   Zone enforcement: OFF (single-subject ~30Q paper)")
        else:
            logger.info(f"   Zone enforcement: OFF (detected {total_questions} questions)")
            
        # Step 3: Analyze each question
        logger.info("Step 3/4: Classifying questions...")
        
        analyzed_questions = []
        
        for eq in extracted_questions:
            try:
                analyzed = self._analyze_single_question(
                    eq, 
                    file_path.name,
                    year=year,
                    shift=shift,
                    total_questions=total_questions,
                    single_section=single_section
                )
                analyzed_questions.append(analyzed)
            except Exception as e:
                warnings.append(f"Failed to analyze Q{eq.question_number}: {str(e)}")
                logger.warning(f"   ⚠️ Q{eq.question_number}: {e}")
                
        logger.info(f"   Classified {len(analyzed_questions)} questions")

        # Attach images to questions using simple proximity by page (round-robin fallback)
        self._attach_images(analyzed_questions, page_images)
        
        # Step 4: Calculate statistics
        logger.info("Step 4/4: Generating statistics...")
        
        section_breakdown = {}
        topic_breakdown = {}
        difficulty_distribution = {"easy": 0, "medium": 0, "hard": 0}
        
        for q in analyzed_questions:
            section_breakdown[q.section] = section_breakdown.get(q.section, 0) + 1
            topic_breakdown[q.topic] = topic_breakdown.get(q.topic, 0) + 1
            difficulty_distribution[q.difficulty] = difficulty_distribution.get(q.difficulty, 0) + 1
            
        result = PaperAnalysisResult(
            source_file=str(file_path),
            year=year,
            shift=shift,
            extraction_date=datetime.now().isoformat(),
            total_questions=len(analyzed_questions),
            questions=analyzed_questions,
            section_breakdown=section_breakdown,
            topic_breakdown=topic_breakdown,
            difficulty_distribution=difficulty_distribution,
            ocr_confidence=avg_confidence,
            extraction_method=extraction_method,
            page_images=page_images,
            warnings=warnings
        )
        
        # Save result
        if save_result:
            self._save_result(result, file_path.stem, year, shift)
            
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _analyze_single_question(
        self,
        question: ExtractedQuestion,
        source_file: str,
        year: Optional[int] = None,
        shift: Optional[int] = None,
        total_questions: int = 100,
        single_section: Optional[str] = None
    ) -> AnalyzedQuestion:
        """Analyze a single extracted question."""
        
        # Handle mathematical formulas
        text = question.text
        has_formula = self.math_handler.has_formula(text)
        if has_formula:
            text = self.math_handler.normalize_math(text)
            
        # Check for diagram references (honor extractor hint first)
        has_diagram = question.has_diagram_reference or bool(__import__('re').search(
            r'figure|diagram|table|graph|chart|above|below',
            text, __import__('re').I
        ))
        
        # Classify topic with question number for zone enforcement
        # Pass question_type for Safe Mode handling of gap-filled placeholders
        topic_result = self.topic_classifier.classify(
            text,
            question_number=question.question_number,
            question_type=question.question_type,  # Safe Mode: zone-based inference
            single_section=single_section
        )
        
        # Classify question type (respect extractor hint if present)
        if question.question_type and question.question_type != "unknown":
            q_type = question.question_type
            type_conf = 0.9
        else:
            q_type, type_conf = self.type_classifier.classify(text)
        
        # Predict difficulty
        difficulty, diff_conf, _ = self.difficulty_predictor.predict(
            text,
            options=question.options,
            topic=topic_result.topic
        )
        
        return AnalyzedQuestion(
            question_number=question.question_number,
            text=text,
            options=question.options,
            section=topic_result.section,
            topic=topic_result.topic,
            topic_label=self.topic_classifier.get_topic_label(topic_result.topic),
            subtopic=topic_result.subtopic,
            question_type=q_type,
            difficulty=difficulty,
            confidence_scores={
                "topic": topic_result.confidence,
                "type": type_conf,
                "difficulty": diff_conf
            },
            has_formula=has_formula,
            has_diagram_reference=has_diagram,
            image_refs=[],
            source_file=source_file,
            confidence=question.confidence,  # Pass extraction confidence
            page_number=question.page_num,
            year=year,
            shift=shift
        )
    
    def _save_result(
        self,
        result: PaperAnalysisResult,
        name: str,
        year: Optional[int],
        shift: Optional[int]
    ):
        """Save analysis result to JSON files."""
        
        # Create filename
        if year and shift:
            filename = f"afcat_{year}_shift{shift}_analysis"
        elif year:
            filename = f"afcat_{year}_analysis"
        else:
            filename = f"{name}_analysis"
            
        # Save full analysis
        full_path = self.output_dir / f"{filename}.json"
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved full analysis: {full_path}")
        
        # Save simplified format for prediction model
        simple_data = self._create_simple_format(result)
        simple_path = self.output_dir / f"{filename}_simple.json"
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved simple format: {simple_path}")
        
        # Save CSV summary
        csv_path = self.output_dir / f"{filename}_topics.csv"
        self._save_topic_csv(result, csv_path)
        logger.info(f"💾 Saved CSV summary: {csv_path}")
        
    def _create_simple_format(self, result: PaperAnalysisResult) -> Dict:
        """Create simplified format for the prediction model."""
        # Group by topic
        topic_counts = {}
        
        for q in result.questions:
            key = (result.year, q.section, q.topic)
            if key not in topic_counts:
                topic_counts[key] = {
                    "year": result.year,
                    "section": q.section,
                    "topic": q.topic,
                    "count": 0,
                    "difficulties": {"easy": 0, "medium": 0, "hard": 0},
                    "sample_questions": []
                }
            
            topic_counts[key]["count"] += 1
            topic_counts[key]["difficulties"][q.difficulty] += 1
            
            # Keep up to 3 sample questions per topic
            if len(topic_counts[key]["sample_questions"]) < 3:
                topic_counts[key]["sample_questions"].append({
                    "text": q.text[:200] + "..." if len(q.text) > 200 else q.text,
                    "difficulty": q.difficulty
                })
                
        return {
            "year": result.year,
            "shift": result.shift,
            "total_questions": result.total_questions,
            "topic_data": list(topic_counts.values()),
            "section_breakdown": result.section_breakdown,
            "difficulty_distribution": result.difficulty_distribution
        }
        
    def _save_topic_csv(self, result: PaperAnalysisResult, path: Path):
        """Save topic breakdown as CSV for easy viewing."""
        rows = ["Year,Section,Topic,Count,Easy,Medium,Hard"]
        
        # Aggregate by topic
        topic_data = {}
        for q in result.questions:
            key = (q.section, q.topic)
            if key not in topic_data:
                topic_data[key] = {"count": 0, "easy": 0, "medium": 0, "hard": 0}
            topic_data[key]["count"] += 1
            topic_data[key][q.difficulty] += 1
            
        for (section, topic), data in sorted(topic_data.items()):
            rows.append(
                f"{result.year or 'Unknown'},{section},{topic},"
                f"{data['count']},{data['easy']},{data['medium']},{data['hard']}"
            )
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows))

    def _attach_images(self, questions: List[AnalyzedQuestion], page_images: Dict[int, List[Dict]]):
        """Attach images to questions heuristically using page grouping and flags."""
        if not page_images:
            return

        # Group questions by page
        page_to_questions: Dict[int, List[AnalyzedQuestion]] = {}
        for q in questions:
            page_to_questions.setdefault(q.page_number, []).append(q)

        for page_num, imgs in page_images.items():
            if not imgs:
                continue
            qs = page_to_questions.get(page_num, [])
            target_qs = [q for q in qs if q.has_diagram_reference or q.question_type in ("non_verbal_figure", "match_following")]
            if not target_qs:
                continue
            img_idx = 0
            for q in target_qs:
                img = imgs[img_idx % len(imgs)]
                q.image_refs = [img]
                q.has_diagram_reference = True
                img_idx += 1
            
    def _print_summary(self, result: PaperAnalysisResult):
        """Print analysis summary to console."""
        print("\n" + "="*60)
        print("📊 ANALYSIS COMPLETE")
        print("="*60)
        print(f"Source: {Path(result.source_file).name}")
        print(f"Year: {result.year or 'Unknown'}, Shift: {result.shift or 'Unknown'}")
        print(f"Total Questions: {result.total_questions}")
        print(f"OCR Confidence: {result.ocr_confidence:.1%}")
        
        print("\n📁 Section Breakdown:")
        for section, count in sorted(result.section_breakdown.items()):
            print(f"   {section}: {count} questions")
            
        print("\n📈 Difficulty Distribution:")
        total = sum(result.difficulty_distribution.values())
        for diff, count in result.difficulty_distribution.items():
            pct = (count / total * 100) if total else 0
            print(f"   {diff.capitalize()}: {count} ({pct:.1f}%)")
            
        print("\n🔝 Top Topics:")
        sorted_topics = sorted(
            result.topic_breakdown.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for topic, count in sorted_topics:
            label = self.topic_classifier.get_topic_label(topic)
            print(f"   {label}: {count}")
            
        if result.warnings:
            print("\n⚠️ Warnings:")
            for w in result.warnings:
                print(f"   - {w}")
                
        print("="*60 + "\n")
        
    def analyze_batch(
        self,
        directory: Union[str, Path],
        file_patterns: List[str] = None,
        years: Optional[Dict[str, int]] = None
    ) -> List[PaperAnalysisResult]:
        """
        Analyze all papers in a directory.
        
        Args:
            directory: Directory containing exam papers
            file_patterns: File patterns to match (default: pdf, png, jpg)
            years: Dict mapping filename to year (optional)
        """
        directory = Path(directory)
        file_patterns = file_patterns or ["*.pdf", "*.png", "*.jpg", "*.jpeg"]
        
        results = []
        
        for pattern in file_patterns:
            for file_path in sorted(directory.glob(pattern)):
                try:
                    # Try to extract year from filename or mapping
                    year = None
                    if years:
                        year = years.get(file_path.name)
                    else:
                        # Try to extract year from filename
                        import re
                        match = re.search(r'20\d{2}', file_path.name)
                        if match:
                            year = int(match.group())
                            
                    result = self.analyze_paper(file_path, year=year)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to analyze {file_path.name}: {e}")
                    
        return results
    
    def get_review_report(self, result: PaperAnalysisResult) -> Dict:
        """Generate report of questions needing review."""
        review_items = []
        
        for q in result.questions:
            check = self.quality_checker.check_question(q)
            if check.needs_review:
                review_items.append({
                    "question_number": q.question_number,
                    "text_preview": q.text[:100] + "..." if len(q.text) > 100 else q.text,
                    "priority": check.priority.value,
                    "issues": check.issues,
                    "suggestions": check.suggestions
                })
                
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        review_items.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return {
            "total_questions": result.total_questions,
            "needs_review": len(review_items),
            "review_rate": len(review_items) / result.total_questions if result.total_questions else 0,
            "items": review_items
        }


# ==================== Convenience Functions ====================

def analyze_exam_paper(
    file_path: str,
    year: Optional[int] = None,
    use_gpu: bool = False
) -> Dict:
    """
    Quick function to analyze an exam paper.
    
    Example:
        result = analyze_exam_paper("afcat_2025_shift1.pdf", year=2025)
        print(f"Found {result['total_questions']} questions")
        print(f"Topics: {result['topic_breakdown']}")
    """
    analyzer = AFCATExamAnalyzer(gpu=use_gpu)
    result = analyzer.analyze_paper(file_path, year=year)
    return result.to_dict()


def batch_analyze_papers(
    directory: str,
    use_gpu: bool = False
) -> List[Dict]:
    """
    Analyze all papers in a directory.
    
    Example:
        results = batch_analyze_papers("papers/2025/")
        for r in results:
            print(f"{r['source_file']}: {r['total_questions']} questions")
    """
    analyzer = AFCATExamAnalyzer(gpu=use_gpu)
    results = analyzer.analyze_batch(directory)
    return [r.to_dict() for r in results]


def convert_to_prediction_format(
    analysis_results: List[Dict],
    output_file: str = "processed_questions.json"
) -> str:
    """
    Convert analysis results to format expected by prediction model.
    
    Creates the Year + Section + Topic + Count format.
    """
    topic_data = []
    
    for result in analysis_results:
        year = result.get("year")
        if not year:
            continue
            
        # Aggregate topics
        topic_counts = {}
        for q in result.get("questions", []):
            key = (year, q["section"], q["topic"])
            if key not in topic_counts:
                topic_counts[key] = 0
            topic_counts[key] += 1
            
        # Convert to list format
        for (y, section, topic), count in topic_counts.items():
            topic_data.append({
                "year": y,
                "section": section,
                "topic": topic,
                "count": count
            })
            
    # Save
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(topic_data, f, indent=2)
        
    return str(output_path)
