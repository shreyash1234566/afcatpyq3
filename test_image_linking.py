"""
Test Image Linking & Figure Detection Accuracy
================================================
Validates OCR quality improvements and neighbor-based figure detection.
"""

import json
import logging
from pathlib import Path
from pipeline.exam_analyzer import AFCATExamAnalyzer, OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_figure_question_markers(question_text):
    """Identify if question likely needs a figure."""
    import re
    
    figure_indicators = [
        r'\b(figure|diagram|image|picture|illustration)\b',
        r'\b(given\s+below|shown\s+below|above\s+figure)\b',
        r'\b(following\s+figure|figure\s+shows)\b',
        r'\b(complete\s+the\s+pattern|next\s+figure|missing\s+figure)\b',
        r'\b(embedded\s+figure|hidden\s+in)\b',
        r'\b(mirror\s+image|water\s+image|reflection)\b',
        r'\b(rotation|rotated|turned)\b',
        r'\b(folded|unfolded|punched)\b',
        r'\b(cube|dice|faces\s+of)\b',
        r'\b(count\s+the|how\s+many).*\b(triangles|squares|lines)\b',
        r'\[FIGURE', r'\[MISSING',  # Our placeholder markers
    ]
    
    text_lower = question_text.lower()
    
    for pattern in figure_indicators:
        if re.search(pattern, text_lower):
            return True
    
    # Suspiciously short questions (< 50 chars) + options = likely figure
    if len(question_text.strip()) < 50:
        has_options = bool(re.search(r'[A-D][\).]', question_text))
        has_minimal_text = len(re.sub(r'[^a-zA-Z]', '', question_text)) < 30
        if has_options and has_minimal_text:
            return True
    
    return False


def test_paper(pdf_path: str, name: str, quality_level: str = "standard"):
    """Test extraction on a single paper with specified quality level."""
    
    print(f"\n{'='*70}")
    print(f"[TEST] Testing: {name}")
    print(f"       Quality: {quality_level}")
    print(f"{'='*70}")
    
    analyzer = AFCATExamAnalyzer(
        ocr_engine=OCREngine.EASYOCR, 
        use_transformers=False, 
        gpu=False,
        quality_level=quality_level
    )
    
    try:
        result = analyzer.analyze_paper(Path(pdf_path), save_result=False)
    except Exception as e:
        print(f"[ERROR] Error analyzing paper: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Statistics
    total_questions = result.total_questions
    questions_flagged_diagram = sum(1 for q in result.questions if q.has_diagram_reference)
    non_verbal_questions = sum(1 for q in result.questions if q.question_type == 'non_verbal_figure')
    match_following = sum(1 for q in result.questions if q.question_type == 'match_following')
    inferred_questions = sum(1 for q in result.questions if q.question_type.startswith('inferred_'))
    
    # Count questions with neighbor context
    questions_with_context = sum(1 for q in result.questions if hasattr(q, 'neighbor_context') and q.neighbor_context)
    
    # Confidence distribution
    high_confidence = sum(1 for q in result.questions if q.confidence >= 0.6)
    medium_confidence = sum(1 for q in result.questions if 0.3 <= q.confidence < 0.6)
    low_confidence = sum(1 for q in result.questions if q.confidence < 0.3)
    
    print(f"\n[EXTRACTION] EXTRACTION SUMMARY:")
    print(f"   Total questions extracted: {total_questions}")
    print(f"   Questions flagged as diagram: {questions_flagged_diagram}")
    print(f"   Non-verbal figure questions: {non_verbal_questions}")
    print(f"   Match-the-following: {match_following}")
    print(f"   Inferred from neighbors: {inferred_questions}")
    
    print(f"\n[CONFIDENCE] CONFIDENCE DISTRIBUTION:")
    print(f"   High (>= 0.6): {high_confidence} ({high_confidence/total_questions*100:.1f}%)")
    print(f"   Medium (0.3-0.6): {medium_confidence} ({medium_confidence/total_questions*100:.1f}%)")
    print(f"   Low (< 0.3): {low_confidence} ({low_confidence/total_questions*100:.1f}%)")
    
    print(f"\n[ANALYSIS] EXPECTED vs ACTUAL:")
    expected_min = int(total_questions * 0.20)
    expected_max = int(total_questions * 0.30)
    actual_figure = non_verbal_questions + match_following
    print(f"   Expected figure/reasoning questions: {expected_min}-{expected_max}")
    print(f"   Actually detected: {actual_figure}")
    match_status = "GOOD" if expected_min <= actual_figure <= expected_max else "CHECK"
    print(f"   Match: {match_status}")
    
    # Question type breakdown
    print(f"\n[BREAKDOWN] QUESTION TYPE DISTRIBUTION:")
    type_breakdown = {}
    for q in result.questions:
        qt = q.question_type
        type_breakdown[qt] = type_breakdown.get(qt, 0) + 1
    
    for qtype, count in sorted(type_breakdown.items(), key=lambda x: -x[1]):
        pct = count / total_questions * 100
        print(f"   {qtype}: {count} ({pct:.1f}%)")
    
    # Sample low-confidence questions for review
    print(f"\n[REVIEW] LOW CONFIDENCE QUESTIONS (samples):")
    low_conf_samples = sorted([q for q in result.questions if q.confidence < 0.4], 
                               key=lambda x: x.confidence)[:5]
    
    for i, q in enumerate(low_conf_samples, 1):
        print(f"\n   {i}. Q{q.question_number} (conf: {q.confidence:.2f})")
        print(f"      Type: {q.question_type}")
        print(f"      Text: {q.text[:60]}...")
        if hasattr(q, 'neighbor_context') and q.neighbor_context:
            ctx = q.neighbor_context
            print(f"      Neighbor hints: [{ctx.prev_topic_hint}, {ctx.next_topic_hint}]")
            print(f"      Inferred section: {ctx.inferred_section}")
    
    # Topic classification breakdown  
    print(f"\n[TOPICS] TOPIC CLASSIFICATION:")
    topic_breakdown = {}
    for q in result.questions:
        topic = q.topic if hasattr(q, 'topic') else 'unknown'
        topic_breakdown[topic] = topic_breakdown.get(topic, 0) + 1
    
    unknown_count = topic_breakdown.get('unknown', 0)
    print(f"   Unknown topics: {unknown_count} ({unknown_count/total_questions*100:.1f}%)")
    print(f"   Top 5 topics:")
    for topic, count in sorted(topic_breakdown.items(), key=lambda x: -x[1])[:5]:
        if topic != 'unknown':
            print(f"      {topic}: {count}")
    
    return {
        'name': name,
        'quality_level': quality_level,
        'total_questions': total_questions,
        'non_verbal_questions': non_verbal_questions,
        'match_following': match_following,
        'questions_flagged_diagram': questions_flagged_diagram,
        'inferred_questions': inferred_questions,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence,
        'unknown_topics': unknown_count,
        'result': result
    }
        'total_page_images': total_page_images,
        'pages_with_images': pages_with_images,
        'result': result
    }


if __name__ == '__main__':
    # Test on 2 sample papers
    papers = [
        ('data/papers/AFCAT_2020_Memory.pdf', '2020 Memory-Based (Baseline)'),
        ('data/papers/AFCAT_2024_Aug10_Shift1_Memory.pdf', '2024 Aug 10 Shift 1 (Recent)'),
    ]
    
    results = []
    
    for pdf_path, name in papers:
        result = test_image_linking(pdf_path, name)
        if result:
            results.append(result)
    
    # Summary comparison
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("[SUMMARY] COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        for r in results:
            rate = (r['questions_with_images'] / r['total_questions'] * 100) if r['total_questions'] > 0 else 0
            print(f"\n{r['name']}:")
            print(f"   Total: {r['total_questions']} | With images: {r['questions_with_images']} ({rate:.1f}%)")
            print(f"   Non-verbal: {r['non_verbal_questions']} | Page images: {r['total_page_images']}")
        
        # Decision
        avg_image_rate = sum(r['questions_with_images'] for r in results) / sum(r['total_questions'] for r in results) * 100
        print(f"\n[DECISION] OVERALL IMAGE LINKING RATE: {avg_image_rate:.1f}%")
        
        if avg_image_rate > 60:
            print(f"   [OK] ROUND-ROBIN IS WORKING REASONABLY WELL")
            print(f"   >> Can proceed with batch extraction")
            print(f"   >> Consider bbox matching only if manual validation shows >20% errors")
        elif avg_image_rate > 30:
            print(f"   [PARTIAL] ROUND-ROBIN IS PARTIAL - NEEDS REFINEMENT")
            print(f"   >> Check if PDFs are extracting images correctly")
            print(f"   >> May need bbox-based matching")
        else:
            print(f"   [FAIL] ROUND-ROBIN NOT CAPTURING IMAGES")
            print(f"   >> PDF image extraction may have failed")
            print(f"   >> Implement bbox-based matching")
    
    print(f"\n{'='*70}\n")
