"""
OCR Preprocessing Module for AFCAT Papers
==========================================
Enhanced image preprocessing for better OCR quality.
Based on best practices from multiple AI analysis sources.

Features:
- Image enhancement (contrast, denoising)
- Deskewing for rotated scans
- Binarization (Otsu's method)
- Question boundary detection
- Missing question number repair
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import image processing libraries
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Image preprocessing disabled.")

try:
    from PIL import Image, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Some preprocessing disabled.")


# ============================================================
# IMAGE ENHANCEMENT FOR OCR
# ============================================================

def enhance_image_for_ocr(image) -> "np.ndarray":
    """
    Improve image quality before OCR.
    Applies: grayscale, denoising, contrast enhancement, binarization.
    
    Args:
        image: numpy array (BGR) or PIL Image
        
    Returns:
        Enhanced numpy array (grayscale, binarized)
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV not available, skipping image enhancement")
        return image
    
    # Convert PIL to numpy if needed
    if PIL_AVAILABLE and isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Binarization using Otsu's method
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def deskew_image(image) -> "np.ndarray":
    """
    Correct rotation/skew in scanned images.
    
    Args:
        image: numpy array (grayscale or color)
        
    Returns:
        Deskewed numpy array
    """
    if not CV2_AVAILABLE:
        return image
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Find coordinates of non-white pixels
    coords = np.column_stack(np.where(gray < 255))
    
    if len(coords) == 0:
        return image
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Skip if angle is very small
    if abs(angle) < 0.5:
        return image
    
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.debug(f"Deskewed image by {angle:.2f} degrees")
    return rotated


def preprocess_page_for_ocr(pil_image: "Image.Image") -> "Image.Image":
    """
    Full preprocessing pipeline for a PDF page image.
    
    Args:
        pil_image: PIL Image of the page
        
    Returns:
        Preprocessed PIL Image
    """
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return pil_image
    
    # Convert to numpy
    img_array = np.array(pil_image.convert('RGB'))
    
    # Deskew
    deskewed = deskew_image(img_array)
    
    # Enhance for OCR
    enhanced = enhance_image_for_ocr(deskewed)
    
    # Convert back to PIL
    return Image.fromarray(enhanced)


# ============================================================
# QUESTION BOUNDARY DETECTION
# ============================================================

# Patterns to detect question numbers (OCR-safe)
QUESTION_NUM_PATTERNS = [
    r'(?:^|\n)\s*[Qq]\.?\s*(\d{1,3})\s*[\.\):\-]',           # Q.1. or Q1) or Q1:
    r'(?:^|\n)\s*[Qq]uestion\s*[Nn]o\.?\s*(\d{1,3})',        # Question No. 1
    r'(?:^|\n)\s*(\d{1,3})\s*[\.\)]\s*(?=[A-Z])',            # 1. or 1) followed by capital
    r'(?:^|\n)\s*\((\d{1,3})\)',                              # (1)
]

# Compiled patterns
COMPILED_Q_PATTERNS = [re.compile(p, re.MULTILINE) for p in QUESTION_NUM_PATTERNS]


def detect_question_boundaries(text: str) -> List[Dict]:
    """
    Detect question start positions in OCR text.
    
    Args:
        text: Full OCR text from a page/document
        
    Returns:
        List of dicts with 'qnum', 'start', 'end' positions
    """
    matches = []
    
    for pattern in COMPILED_Q_PATTERNS:
        for match in pattern.finditer(text):
            try:
                qnum = int(match.group(1))
                if 1 <= qnum <= 200:  # Valid question range
                    matches.append({
                        'qnum': qnum,
                        'start': match.start(),
                        'end': match.end(),
                        'matched_text': match.group(0)
                    })
            except (ValueError, IndexError):
                continue
    
    # Sort by position and remove duplicates
    matches.sort(key=lambda x: x['start'])
    
    # Remove overlapping matches (keep first)
    cleaned = []
    last_end = -1
    for m in matches:
        if m['start'] >= last_end:
            cleaned.append(m)
            last_end = m['end']
    
    return cleaned


def split_into_questions(text: str, expected_total: int = 100) -> List[Dict]:
    """
    Split OCR text into individual questions.
    
    Args:
        text: Full OCR text
        expected_total: Expected number of questions (for gap detection)
        
    Returns:
        List of dicts with 'qnum', 'text' for each question
    """
    boundaries = detect_question_boundaries(text)
    
    if not boundaries:
        logger.warning("No question boundaries detected, using fallback")
        # Fallback: split on double newlines
        chunks = re.split(r'\n\s*\n', text)
        return [{'qnum': None, 'text': c.strip()} for c in chunks if c.strip()]
    
    questions = []
    for i, boundary in enumerate(boundaries):
        start = boundary['start']
        end = boundaries[i + 1]['start'] if i + 1 < len(boundaries) else len(text)
        
        qtext = text[start:end].strip()
        questions.append({
            'qnum': boundary['qnum'],
            'text': qtext,
            'confidence': 1.0  # High confidence for detected boundaries
        })
    
    return questions


def repair_missing_numbers(
    questions: List[Dict],
    expected_total: int = 100
) -> Tuple[List[Dict], List[int]]:
    """
    Repair missing question numbers by inferring from sequence.
    
    Args:
        questions: List of question dicts
        expected_total: Expected total questions
        
    Returns:
        Tuple of (repaired questions, list of missing numbers)
    """
    # Get all detected numbers
    detected = set(q['qnum'] for q in questions if q.get('qnum'))
    expected = set(range(1, expected_total + 1))
    missing = sorted(expected - detected)
    
    if not missing:
        return questions, []
    
    logger.info(f"Detected {len(detected)} questions, missing: {len(missing)}")
    
    # If questions have None qnum, try to assign from missing
    repaired = []
    missing_iter = iter(missing)
    
    for q in questions:
        if q.get('qnum') is None:
            try:
                q['qnum'] = next(missing_iter)
                q['confidence'] = 0.5  # Lower confidence for inferred
            except StopIteration:
                pass
        repaired.append(q)
    
    # Sort by question number
    repaired.sort(key=lambda x: x.get('qnum', 999))
    
    # Get remaining missing (not assigned)
    still_missing = [n for n in missing if n not in {q.get('qnum') for q in repaired}]
    
    return repaired, still_missing


# ============================================================
# QUESTION QUALITY SCORING
# ============================================================

@dataclass
class QuestionQuality:
    """Quality assessment for an extracted question."""
    has_options: bool
    has_question_mark: bool
    text_length: int
    ocr_confidence: float
    has_directive_cue: bool
    quality_score: float


# Directive cues that indicate a real question
DIRECTIVE_CUES = [
    r'choose\s+the\s+correct',
    r'select\s+the\s+(?:correct|right|best)',
    r'find\s+(?:the|out)',
    r'which\s+(?:of|one)',
    r'what\s+(?:is|are|was|were)',
    r'who\s+(?:is|are|was|were)',
    r'where\s+(?:is|are|was|were)',
    r'when\s+(?:is|was|did)',
    r'how\s+(?:many|much|is|are)',
    r'identify\s+the',
    r'solve\s+(?:the|for)',
    r'calculate',
    r'simplify',
    r'answer\s+figure',
    r'read\s+the\s+passage',
]

COMPILED_CUES = [re.compile(c, re.IGNORECASE) for c in DIRECTIVE_CUES]


def compute_question_quality(
    question: Dict,
    ocr_confidence: float = 0.7
) -> QuestionQuality:
    """
    Compute quality score for an extracted question.
    Used to gate zone enforcement (don't enforce on low-quality placeholders).
    
    Args:
        question: Dict with 'text', 'options' (optional)
        ocr_confidence: OCR confidence score (0-1)
        
    Returns:
        QuestionQuality dataclass
    """
    text = question.get('text', '')
    options = question.get('options', [])
    
    # Check for options (a/b/c/d patterns)
    has_options = len(options) >= 2 or bool(
        re.search(r'\([abcd]\)|\b[abcd]\s*[\.:\)]', text, re.IGNORECASE)
    )
    
    # Check for question mark
    has_question_mark = '?' in text
    
    # Text length
    text_length = len(text.strip())
    
    # Check for directive cues
    has_directive = any(p.search(text) for p in COMPILED_CUES)
    
    # Compute quality score (0-1)
    score = 0.0
    
    # Options detected (+0.3)
    if has_options:
        score += 0.30
    
    # Question mark (+0.15)
    if has_question_mark:
        score += 0.15
    
    # Sufficient text length (+0.2 if > 30 chars, +0.1 if > 15)
    if text_length > 50:
        score += 0.20
    elif text_length > 30:
        score += 0.15
    elif text_length > 15:
        score += 0.10
    
    # Directive cue (+0.2)
    if has_directive:
        score += 0.20
    
    # OCR confidence factor (+0.15 max)
    score += 0.15 * min(ocr_confidence, 1.0)
    
    return QuestionQuality(
        has_options=has_options,
        has_question_mark=has_question_mark,
        text_length=text_length,
        ocr_confidence=ocr_confidence,
        has_directive_cue=has_directive,
        quality_score=min(score, 1.0)
    )


def is_placeholder_question(question: Dict, quality_threshold: float = 0.35) -> bool:
    """
    Determine if a question is likely a placeholder (not real extracted content).
    
    Args:
        question: Question dict
        quality_threshold: Below this score, consider placeholder
        
    Returns:
        True if likely a placeholder
    """
    text = question.get('text', '').lower()
    
    # Explicit placeholder markers
    placeholder_markers = [
        'placeholder',
        'question not available',
        'text not extracted',
        'ocr failed',
        '[verbal',
        '[ga',
        '[reasoning',
        '[math',
        'trailing placeholder',
        'inferred'
    ]
    
    if any(marker in text for marker in placeholder_markers):
        return True
    
    # Check quality score
    quality = compute_question_quality(question)
    if quality.quality_score < quality_threshold:
        return True
    
    # Very short text without options
    if len(text) < 15 and not quality.has_options:
        return True
    
    return False


# ============================================================
# ZONE ENFORCEMENT WITH QUALITY GATING
# ============================================================

AFCAT_ZONE_DISTRIBUTION = {
    'verbal_ability': (1, 30, 30),       # (start, end, count)
    'general_awareness': (31, 55, 25),
    'reasoning': (56, 80, 25),
    'numerical_ability': (81, 100, 20),
}


def enforce_zone_counts(
    questions: List[Dict],
    expected_counts: Dict[str, int] = None,
    quality_threshold: float = 0.40
) -> Tuple[List[Dict], List[Dict]]:
    """
    Enforce zone-based section distribution with quality gating.
    Only reassigns questions that meet quality threshold.
    
    Args:
        questions: List of classified question dicts
        expected_counts: Override expected counts per section
        quality_threshold: Minimum quality to apply zone enforcement
        
    Returns:
        Tuple of (updated questions, reassignment log)
    """
    if expected_counts is None:
        expected_counts = {
            'verbal_ability': 30,
            'general_awareness': 25,
            'reasoning': 25,
            'numerical_ability': 20
        }
    
    reassignments = []
    
    for q in questions:
        qnum = q.get('question_number', q.get('qnum', 0))
        current_section = q.get('section', 'unknown')
        
        # Determine expected section from zone
        expected_section = None
        for section, (start, end, _) in AFCAT_ZONE_DISTRIBUTION.items():
            if start <= qnum <= end:
                expected_section = section
                break
        
        if expected_section is None:
            continue
        
        # Check if reassignment needed
        if current_section != expected_section:
            # Quality gate: only reassign if quality is above threshold
            quality = compute_question_quality(q)
            
            if quality.quality_score >= quality_threshold:
                # High quality: trust keyword classification more
                # But still log the mismatch
                reassignments.append({
                    'qnum': qnum,
                    'from': current_section,
                    'to': expected_section,
                    'action': 'kept_original',
                    'reason': 'high_quality_keyword_match',
                    'quality_score': quality.quality_score
                })
            else:
                # Low quality: force zone assignment
                old_section = current_section
                q['section'] = expected_section
                q['method'] = q.get('method', '') + '+zone-forced'
                
                reassignments.append({
                    'qnum': qnum,
                    'from': old_section,
                    'to': expected_section,
                    'action': 'reassigned',
                    'reason': 'low_quality_zone_enforced',
                    'quality_score': quality.quality_score
                })
    
    return questions, reassignments


# ============================================================
# DIAGNOSTIC UTILITIES
# ============================================================

def diagnose_extraction(
    questions: List[Dict],
    expected_total: int = 100
) -> Dict:
    """
    Diagnose extraction quality and identify issues.
    
    Args:
        questions: List of extracted question dicts
        expected_total: Expected number of questions
        
    Returns:
        Diagnostic report dict
    """
    q_numbers = [q.get('question_number', q.get('qnum', 0)) for q in questions]
    detected = set(q_numbers)
    expected = set(range(1, expected_total + 1))
    missing = sorted(expected - detected)
    
    # Confidence distribution
    confidences = [q.get('confidence', 0.5) for q in questions]
    conf_buckets = {
        'high (>80%)': sum(1 for c in confidences if c > 0.8),
        'medium (60-80%)': sum(1 for c in confidences if 0.6 <= c <= 0.8),
        'low (<60%)': sum(1 for c in confidences if c < 0.6)
    }
    
    # Placeholder count
    placeholder_count = sum(1 for q in questions if is_placeholder_question(q))
    
    # Section distribution
    sections = {}
    for q in questions:
        sec = q.get('section', 'unknown')
        sections[sec] = sections.get(sec, 0) + 1
    
    report = {
        'total_extracted': len(questions),
        'expected': expected_total,
        'missing_count': len(missing),
        'missing_numbers': missing[:20],  # First 20 only
        'confidence_distribution': conf_buckets,
        'placeholder_count': placeholder_count,
        'real_question_count': len(questions) - placeholder_count,
        'section_distribution': sections,
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
    }
    
    return report


def print_diagnostic_report(report: Dict):
    """Print a formatted diagnostic report."""
    print("\n" + "=" * 60)
    print("📊 EXTRACTION DIAGNOSTIC REPORT")
    print("=" * 60)
    
    print(f"\n📋 Question Counts:")
    print(f"   Total extracted: {report['total_extracted']}")
    print(f"   Expected: {report['expected']}")
    print(f"   Missing: {report['missing_count']}")
    print(f"   Placeholders: {report['placeholder_count']}")
    print(f"   Real questions: {report['real_question_count']}")
    
    if report['missing_numbers']:
        print(f"\n⚠️  Missing question numbers: {report['missing_numbers']}")
    
    print(f"\n📈 Confidence Distribution:")
    for bucket, count in report['confidence_distribution'].items():
        print(f"   {bucket}: {count}")
    print(f"   Average confidence: {report['avg_confidence']:.1%}")
    
    print(f"\n📁 Section Distribution:")
    for section, count in report['section_distribution'].items():
        expected = AFCAT_ZONE_DISTRIBUTION.get(section, (0, 0, 0))[2]
        status = "✓" if count == expected else "⚠️"
        print(f"   {status} {section}: {count} (expected: {expected})")
    
    print("\n" + "=" * 60)
