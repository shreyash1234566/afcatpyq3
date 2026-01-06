#!/usr/bin/env python3
"""
Hybrid Classification System for AFCAT Questions.

Combines API-based topic detection with sequential section enforcement.
Key features:
1. API classifies questions by content (topic + initial section)
2. Sequential enforcer ensures section order: VA → GA → NA → RM
3. Auto-detects paper type (100Q full vs 30Q single-subject)
4. Dynamic boundary detection from API result clustering
5. Majority vote for single-subject papers

Author: AFCAT Prediction System
"""

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import requests

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Section order for AFCAT (sequential, never goes back)
SECTION_ORDER = ["verbal_ability", "general_awareness", "numerical_ability", "reasoning"]

# Section short codes for display
SECTION_SHORT = {
    "verbal_ability": "VA",
    "general_awareness": "GA",
    "numerical_ability": "NA",
    "reasoning": "RM",
    "unknown": "??"
}

# Paper type thresholds
FULL_PAPER_MIN_QUESTIONS = 80
SINGLE_SUBJECT_MAX_QUESTIONS = 40

# Confidence threshold: API wins if >= this, zone wins otherwise
API_CONFIDENCE_THRESHOLD = 0.90

# Rate limiting for Groq API (free tier ~30 RPM)
API_DELAY_SECONDS = 2.0
API_RETRY_WAIT = 10

# Default zones (used as initial hints, can be overridden by boundary detection)
DEFAULT_ZONES = {
    "verbal_ability": (1, 30),
    "general_awareness": (31, 55),
    "numerical_ability": (56, 75),
    "reasoning": (76, 100),
}

# Topic codes by section
TOPIC_CODES = {
    "verbal_ability": [
        ("VA_SYN", "Synonyms"),
        ("VA_ANT", "Antonyms"),
        ("VA_IDIOM", "Idioms & Phrases"),
        ("VA_OWS", "One Word Substitution"),
        ("VA_COMP", "Reading Comprehension"),
        ("VA_CLOZE", "Cloze Test"),
        ("VA_PARA", "Para Jumbles"),
        ("VA_ERR", "Error Spotting"),
        ("VA_FILL", "Fill in the Blanks"),
        ("VA_SENT", "Sentence Improvement"),
    ],
    "general_awareness": [
        ("GA_HIST_ANC", "Ancient History"),
        ("GA_HIST_MED", "Medieval History"),
        ("GA_HIST_MOD", "Modern History"),
        ("GA_GEO_IND", "Indian Geography"),
        ("GA_GEO_WORLD", "World Geography"),
        ("GA_POLITY", "Polity & Governance"),
        ("GA_ECON", "Economy"),
        ("GA_SCI", "Science & Technology"),
        ("GA_DEF", "Defence & Military"),
        ("GA_CURR", "Current Affairs"),
        ("GA_SPORTS", "Sports"),
        ("GA_AWARDS", "Awards & Honours"),
        ("GA_CULTURE", "Art & Culture"),
        ("GA_ORG", "Organizations"),
    ],
    "numerical_ability": [
        ("NA_NUM", "Number System"),
        ("NA_PER", "Percentage"),
        ("NA_RAT", "Ratio & Proportion"),
        ("NA_AVG", "Average"),
        ("NA_PL", "Profit & Loss"),
        ("NA_SI", "Simple Interest"),
        ("NA_CI", "Compound Interest"),
        ("NA_TW", "Time & Work"),
        ("NA_SPD", "Speed, Distance, Time"),
        ("NA_TRAIN", "Trains & Platforms"),
        ("NA_DI", "Data Interpretation"),
        ("NA_ALG", "Algebra"),
        ("NA_GEO", "Geometry & Mensuration"),
    ],
    "reasoning": [
        ("RM_VR_ANALOGY", "Verbal Analogy"),
        ("RM_VR_SERIES", "Verbal Series"),
        ("RM_VR_CLASS", "Verbal Classification"),
        ("RM_VR_CODING", "Coding-Decoding"),
        ("RM_VR_DIR", "Direction Sense"),
        ("RM_VR_BLOOD", "Blood Relations"),
        ("RM_VR_ORDER", "Order & Ranking"),
        ("RM_NV_ANALOGY", "Non-Verbal Analogy"),
        ("RM_NV_SERIES", "Non-Verbal Series"),
        ("RM_NV_CLASS", "Non-Verbal Classification"),
        ("RM_NV_PATTERN", "Pattern Completion"),
        ("RM_NV_FIG", "Figure-based Questions"),
        ("RM_NV_MIRROR", "Mirror & Water Images"),
        ("RM_NV_VENN", "Venn Diagrams"),
        ("RM_NV_CUBES", "Cubes & Dice"),
    ],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class QuestionClassification:
    """Result of classifying a single question."""
    question_number: int
    section: str
    topic_code: str
    topic_name: str
    confidence: float
    method: str  # "api", "api+sequential", "api+zone", "majority", etc.
    original_section: Optional[str] = None  # Before enforcement
    text: str = ""
    options: List[str] = field(default_factory=list)


@dataclass
class PaperClassification:
    """Result of classifying an entire paper."""
    paper_type: str  # "full" or "single_subject"
    total_questions: int
    detected_section_order: List[str]
    detected_boundaries: Dict[str, Tuple[int, int]]
    questions: List[QuestionClassification]
    section_counts: Dict[str, int]
    topic_counts: Dict[str, int]
    accuracy_estimate: float  # Based on confidence scores
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# HYBRID CLASSIFIER
# ============================================================================

class HybridClassifier:
    """
    Hybrid classification system combining API + Sequential Enforcement.
    
    Pipeline:
    1. Detect paper type (full 100Q vs single-subject 30Q)
    2. API classifies all questions (topic + initial section)
    3. Sequential enforcer fixes impossible jumps (VA→GA→NA→RM order)
    4. Boundary detection validates section transitions
    5. Final output with corrected sections
    """
    
    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = GROQ_MODEL,
        confidence_threshold: float = API_CONFIDENCE_THRESHOLD,
        verbose: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.detected_order = list(SECTION_ORDER)  # Will be updated after API pass
        
    def classify_paper(
        self,
        questions: List[Dict[str, Any]],
        force_single_section: Optional[str] = None
    ) -> PaperClassification:
        """
        Main entry point: classify all questions in a paper.
        
        Args:
            questions: List of dicts with 'question_number', 'text', 'options'
            force_single_section: If set, override all questions to this section
            
        Returns:
            PaperClassification with all results
        """
        total = len(questions)
        warnings = []
        
        # Step 1: Detect paper type
        paper_type = self._detect_paper_type(total)
        if self.verbose:
            print(f"\n📋 Paper Type: {paper_type.upper()} ({total} questions)")
        
        # Step 2: API classification for all questions
        if self.verbose:
            print(f"\n🤖 Step 1: API Classification ({self.model})...")
        api_results = self._api_classify_all(questions)
        
        # Step 3: Handle based on paper type
        if force_single_section:
            # User forced single section
            if self.verbose:
                print(f"\n🎯 Forcing all questions to: {force_single_section}")
            final_results = self._apply_single_section(api_results, force_single_section)
            paper_type = "single_subject"
            
        elif paper_type == "single_subject":
            # Auto-detect section via majority vote
            if self.verbose:
                print(f"\n📊 Step 2: Majority Vote for Single-Subject...")
            final_results, detected_section = self._apply_majority_vote(api_results)
            if self.verbose:
                print(f"   ✓ Detected section: {detected_section}")
                
        else:
            # Full paper: apply sequential enforcement
            if self.verbose:
                print(f"\n🔒 Step 2: Sequential Section Enforcement...")
            
            # Detect section order from API results
            self.detected_order = self._detect_section_order(api_results)
            if self.verbose:
                order_str = " → ".join([SECTION_SHORT.get(s, s) for s in self.detected_order])
                print(f"   Detected order: {order_str}")
            
            # Apply sequential enforcement
            final_results = self._apply_sequential_enforcement(api_results)
            
            # Detect boundaries
            boundaries = self._detect_boundaries(final_results)
            if self.verbose:
                print(f"\n📍 Step 3: Boundary Detection...")
                for section, (start, end) in boundaries.items():
                    print(f"   {SECTION_SHORT.get(section, section)}: Q{start}-Q{end}")
        
        # Calculate section and topic counts
        section_counts = Counter(q.section for q in final_results)
        topic_counts = Counter(q.topic_name for q in final_results)
        
        # Estimate accuracy from confidence scores
        avg_confidence = sum(q.confidence for q in final_results) / len(final_results) if final_results else 0
        
        # Build result
        boundaries = self._detect_boundaries(final_results) if paper_type == "full" else {}
        
        result = PaperClassification(
            paper_type=paper_type,
            total_questions=total,
            detected_section_order=self.detected_order,
            detected_boundaries=boundaries,
            questions=final_results,
            section_counts=dict(section_counts),
            topic_counts=dict(topic_counts),
            accuracy_estimate=avg_confidence,
            warnings=warnings
        )
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    # ========================================================================
    # PAPER TYPE DETECTION
    # ========================================================================
    
    def _detect_paper_type(self, question_count: int) -> str:
        """Detect if this is a full paper or single-subject paper."""
        if question_count >= FULL_PAPER_MIN_QUESTIONS:
            return "full"
        elif question_count <= SINGLE_SUBJECT_MAX_QUESTIONS:
            return "single_subject"
        else:
            # Ambiguous range (41-79 questions)
            logger.warning(f"Ambiguous question count: {question_count}")
            return "full" if question_count >= 60 else "single_subject"
    
    # ========================================================================
    # API CLASSIFICATION
    # ========================================================================
    
    def _api_classify_all(self, questions: List[Dict]) -> List[QuestionClassification]:
        """Classify all questions via Groq API."""
        results = []
        
        for i, q in enumerate(questions):
            qnum = q.get("question_number", i + 1)
            qtext = q.get("text", "")[:500]  # Limit text length
            options = q.get("options", [])
            
            # Rate limiting
            if i > 0:
                time.sleep(API_DELAY_SECONDS)
            
            # API call with retry
            result = self._api_classify_single(qnum, qtext, options)
            results.append(result)
            
            if self.verbose:
                sec = SECTION_SHORT.get(result.section, "??")
                print(f"   ✓ Q{qnum:>3}: {sec:<4} | {result.topic_code:<16} | {result.topic_name[:25]}")
        
        return results
    
    def _api_classify_single(
        self,
        question_number: int,
        text: str,
        options: List[str]
    ) -> QuestionClassification:
        """Classify a single question via Groq API."""
        
        # Build prompt
        options_text = "\n".join([f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options[:4])])
        
        prompt = f"""Classify this AFCAT exam question.

Question {question_number}:
{text}
{options_text}

Return JSON with:
- section: one of [verbal_ability, general_awareness, numerical_ability, reasoning]
- topic_code: e.g., VA_SYN, GA_HIST_MOD, NA_PER, RM_VR_ANALOGY
- topic_name: e.g., "Synonyms", "Modern History", "Percentage", "Verbal Analogy"
- confidence: 0.0-1.0

JSON only:"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an AFCAT exam question classifier. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        try:
            # Retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429 and attempt < max_retries - 1:
                        wait_time = API_RETRY_WAIT * (attempt + 1)
                        if self.verbose:
                            print(f"   ⏳ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            result = response.json()
            text_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Parse JSON from response
            text_response = re.sub(r'^```json\s*', '', text_response)
            text_response = re.sub(r'\s*```$', '', text_response)
            
            json_match = re.search(r'\{[^}]+\}', text_response)
            if json_match:
                data = json.loads(json_match.group())
                return QuestionClassification(
                    question_number=question_number,
                    section=data.get("section", "unknown"),
                    topic_code=data.get("topic_code", "unknown"),
                    topic_name=data.get("topic_name", "unknown"),
                    confidence=float(data.get("confidence", 0.5)),
                    method="api",
                    text=text,
                    options=options
                )
        
        except Exception as e:
            logger.error(f"API error for Q{question_number}: {e}")
            if self.verbose:
                print(f"   ⚠️ Q{question_number}: API error - {str(e)[:50]}")
        
        # Fallback for errors
        return QuestionClassification(
            question_number=question_number,
            section="unknown",
            topic_code="unknown",
            topic_name="unknown",
            confidence=0.0,
            method="api-error",
            text=text,
            options=options
        )
    
    # ========================================================================
    # SECTION ORDER DETECTION
    # ========================================================================
    
    def _detect_section_order(self, results: List[QuestionClassification]) -> List[str]:
        """
        Detect the actual section order from API results.
        
        AFCAT typically follows VA → GA → NA → RM or VA → GA → RM → NA.
        This function detects which order by looking at where each section
        FIRST APPEARS AS A CLUSTER (5+ consecutive questions).
        
        Single outlier questions (like Q3 returning NA when surrounded by VA)
        should NOT trigger a section transition.
        """
        sorted_results = sorted(results, key=lambda x: x.question_number)
        
        # Find first occurrence of 5+ consecutive questions in each section
        MIN_CLUSTER_SIZE = 5
        first_cluster = {}
        
        for section in SECTION_ORDER:
            # Find runs of this section
            run_start = None
            run_length = 0
            
            for q in sorted_results:
                if q.section == section and q.confidence >= 0.6:
                    if run_start is None:
                        run_start = q.question_number
                        run_length = 1
                    else:
                        run_length += 1
                else:
                    # Run broken - check if it was long enough
                    if run_length >= MIN_CLUSTER_SIZE:
                        if section not in first_cluster:
                            first_cluster[section] = run_start
                        break
                    run_start = None
                    run_length = 0
            
            # Check final run
            if run_length >= MIN_CLUSTER_SIZE and section not in first_cluster:
                first_cluster[section] = run_start
        
        # Sort sections by first cluster occurrence
        detected = sorted(first_cluster.keys(), key=lambda s: first_cluster.get(s, 999))
        
        # Fill in any missing sections using standard AFCAT order
        # Standard order: VA → GA → NA → RM (but NA/RM can be swapped)
        for section in SECTION_ORDER:
            if section not in detected:
                detected.append(section)
        
        return detected
    
    # ========================================================================
    # SEQUENTIAL ENFORCEMENT
    # ========================================================================
    
    def _apply_sequential_enforcement(
        self,
        results: List[QuestionClassification]
    ) -> List[QuestionClassification]:
        """
        Enforce sequential section order: once you leave a section, you NEVER go back.
        
        Example: If Q50 is GA, Q51 can be GA/NA/RM but NEVER VA.
        
        Algorithm:
        1. Sort questions by number
        2. Track current "minimum allowed section index"
        3. If API returns earlier section, override to current minimum
        4. If API returns later section with high confidence, update minimum
        """
        sorted_results = sorted(results, key=lambda x: x.question_number)
        section_index = {s: i for i, s in enumerate(self.detected_order)}
        
        current_min_idx = 0  # Can't go before this section
        enforced_results = []
        
        for q in sorted_results:
            original_section = q.section
            q.original_section = original_section
            
            if q.section == "unknown" or q.section not in section_index:
                # Unknown section: use current expected section
                q.section = self.detected_order[current_min_idx]
                q.method = "api+inferred"
                enforced_results.append(q)
                continue
            
            api_section_idx = section_index.get(q.section, 0)
            
            if api_section_idx < current_min_idx:
                # API says earlier section - IMPOSSIBLE, override
                q.section = self.detected_order[current_min_idx]
                q.method = "api+sequential"
            elif api_section_idx > current_min_idx:
                # API says later section
                if q.confidence >= self.confidence_threshold:
                    # High confidence: trust API, update minimum
                    current_min_idx = api_section_idx
                    q.method = "api"
                else:
                    # Low confidence: stay in current section
                    q.section = self.detected_order[current_min_idx]
                    q.method = "api+zone"
            else:
                # Same section: keep as-is
                q.method = "api"
            
            enforced_results.append(q)
        
        return enforced_results
    
    # ========================================================================
    # BOUNDARY DETECTION
    # ========================================================================
    
    def _detect_boundaries(
        self,
        results: List[QuestionClassification]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Detect actual section boundaries from classified results.
        
        Returns dict like:
        {
            "verbal_ability": (1, 25),
            "general_awareness": (26, 50),
            ...
        }
        """
        sorted_results = sorted(results, key=lambda x: x.question_number)
        boundaries = {}
        
        current_section = None
        section_start = None
        
        for q in sorted_results:
            if q.section != current_section:
                # Section changed
                if current_section is not None:
                    boundaries[current_section] = (section_start, q.question_number - 1)
                current_section = q.section
                section_start = q.question_number
        
        # Don't forget the last section
        if current_section is not None and sorted_results:
            boundaries[current_section] = (section_start, sorted_results[-1].question_number)
        
        return boundaries
    
    # ========================================================================
    # SINGLE-SUBJECT HANDLING
    # ========================================================================
    
    def _apply_majority_vote(
        self,
        results: List[QuestionClassification]
    ) -> Tuple[List[QuestionClassification], str]:
        """
        For single-subject papers, use majority vote to determine section.
        All questions get assigned to the most common section.
        """
        # Count sections
        section_counts = Counter(q.section for q in results if q.section != "unknown")
        
        if not section_counts:
            # All unknown - default to verbal_ability
            majority_section = "verbal_ability"
        else:
            majority_section = section_counts.most_common(1)[0][0]
            majority_count = section_counts[majority_section]
            total = sum(section_counts.values())
            margin = majority_count / total if total > 0 else 0
            
            if margin < 0.6 and self.verbose:
                print(f"   ⚠️ Low confidence: {majority_section} has only {margin:.0%} majority")
        
        # Apply to all questions
        for q in results:
            q.original_section = q.section
            q.section = majority_section
            q.method = "api+majority"
        
        return results, majority_section
    
    def _apply_single_section(
        self,
        results: List[QuestionClassification],
        section: str
    ) -> List[QuestionClassification]:
        """Force all questions to a specific section."""
        for q in results:
            q.original_section = q.section
            q.section = section
            q.method = "forced"
        return results
    
    # ========================================================================
    # OUTPUT & REPORTING
    # ========================================================================
    
    def _print_summary(self, result: PaperClassification):
        """Print classification summary."""
        print("\n" + "=" * 70)
        print("📊 CLASSIFICATION SUMMARY")
        print("=" * 70)
        
        print(f"\n📁 Section Breakdown:")
        for section in self.detected_order:
            count = result.section_counts.get(section, 0)
            pct = (count / result.total_questions * 100) if result.total_questions > 0 else 0
            print(f"   {SECTION_SHORT.get(section, section):<4} ({section:<20}): {count:>3} questions ({pct:>5.1f}%)")
        
        if result.paper_type == "full" and result.detected_boundaries:
            print(f"\n📍 Detected Boundaries:")
            for section, (start, end) in result.detected_boundaries.items():
                print(f"   {SECTION_SHORT.get(section, section)}: Q{start} - Q{end} ({end - start + 1} questions)")
        
        print(f"\n🔝 Top Topics:")
        for topic, count in sorted(result.topic_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"   {topic:<35}: {count:>3}")
        
        print(f"\n📈 Average Confidence: {result.accuracy_estimate:.1%}")
        print("=" * 70)
    
    def export_json(self, result: PaperClassification, output_path: str):
        """Export results to JSON file."""
        data = {
            "paper_type": result.paper_type,
            "total_questions": result.total_questions,
            "detected_section_order": result.detected_section_order,
            "detected_boundaries": {k: list(v) for k, v in result.detected_boundaries.items()},
            "section_breakdown": result.section_counts,
            "topic_breakdown": result.topic_counts,
            "accuracy_estimate": result.accuracy_estimate,
            "questions": [
                {
                    "question_number": q.question_number,
                    "section": q.section,
                    "topic_code": q.topic_code,
                    "topic_name": q.topic_name,
                    "confidence": q.confidence,
                    "method": q.method,
                    "original_section": q.original_section,
                    "text": q.text[:200],
                    "options": q.options[:4]
                }
                for q in result.questions
            ],
            "warnings": result.warnings
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n💾 Saved to: {output_path}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point for hybrid classification."""
    import argparse
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    parser = argparse.ArgumentParser(description="Hybrid AFCAT Question Classifier")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--year", type=int, default=2025, help="Exam year")
    parser.add_argument("--shift", type=int, default=1, help="Shift number")
    parser.add_argument("--single-section", help="Force single section (e.g., verbal_ability)")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Extract questions from PDF
    from utils.ocr_engine import ExamPaperOCR, MCQExtractor, OCREngine
    
    print("=" * 70)
    print("HYBRID CLASSIFICATION SYSTEM")
    print("=" * 70)
    print(f"\n📄 PDF: {args.pdf_path}")
    print(f"🤖 Model: {GROQ_MODEL}")
    print("-" * 70)
    
    print("\n📖 Extracting questions from PDF...")
    ocr = ExamPaperOCR(engine=OCREngine.EASYOCR)
    ocr_results = ocr.extract_from_pdf(args.pdf_path)
    full_text = ocr.get_full_text(ocr_results)
    
    mcq_extractor = MCQExtractor()
    raw_questions = mcq_extractor.extract_questions(full_text)
    
    print(f"   ✓ Extracted {len(raw_questions)} questions")
    
    # Convert to dict format
    questions = [
        {
            "question_number": q.question_number,
            "text": q.text,
            "options": q.options
        }
        for q in raw_questions
    ]
    
    # Run hybrid classifier
    classifier = HybridClassifier(verbose=not args.quiet)
    result = classifier.classify_paper(questions, force_single_section=args.single_section)
    
    # Export
    if args.output:
        output_path = args.output
    else:
        import os
        os.makedirs("output/hybrid_classification", exist_ok=True)
        output_path = f"output/hybrid_classification/afcat_{args.year}_{args.shift}_hybrid.json"
    
    classifier.export_json(result, output_path)
    
    print("\n" + "=" * 70)
    print("✅ Hybrid classification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
