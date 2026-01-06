"""
Test OCR Quality & Figure Detection Improvements
==================================================
Validates:
1. OCR quality improvements
2. Neighbor-based figure inference with confidence
3. Content-based detection (no filename assumptions)
"""

import json
import logging
from pathlib import Path
from pipeline.exam_analyzer import AFCATExamAnalyzer, OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_figure_markers(question_text):
    """Identify if question likely needs a figure."""
    import re
    
    figure_indicators = [
        r'\b(figure|diagram|image|picture|illustration)\b',
        r'\b(given\s+below|shown\s+below|above\s+figure)\b',
        r'\b(complete\s+the\s+pattern|next\s+figure|missing\s+figure)\b',
        r'\b(embedded\s+figure|hidden\s+in)\b',
        r'\b(mirror\s+image|water\s+image|reflection)\b',
        r'\b(rotation|rotated|turned)\b',
        r'\b(folded|unfolded|punched)\b',
        r'\b(cube|dice|faces\s+of)\b',
        r'\b(count\s+the|how\s+many).*\b(triangles|squares|lines)\b',
        r'\[FIGURE', r'\[MISSING',
    ]
    
    text_lower = question_text.lower()
    for pattern in figure_indicators:
        if re.search(pattern, text_lower):
            return True
    return False


def test_paper(pdf_path: str, name: str):
    """Test extraction on a single paper."""
    
    print(f"\n{'='*70}")
    print(f"[TEST] Testing: {name}")
    print(f"{'='*70}")
    
    if not Path(pdf_path).exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return None
    
    analyzer = AFCATExamAnalyzer(
        ocr_engine=OCREngine.EASYOCR, 
        use_transformers=False, 
        gpu=False
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
    if total_questions > 0:
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
        pct = count / total_questions * 100 if total_questions > 0 else 0
        print(f"   {qtype}: {count} ({pct:.1f}%)")
    
    # Sample low-confidence questions for review
    print(f"\n[REVIEW] LOW CONFIDENCE QUESTIONS (samples):")
    low_conf_samples = sorted([q for q in result.questions if q.confidence < 0.4], 
                               key=lambda x: x.confidence)[:5]
    
    for i, q in enumerate(low_conf_samples, 1):
        print(f"\n   {i}. Q{q.question_number} (conf: {q.confidence:.2f})")
        print(f"      Type: {q.question_type}")
        text_preview = q.text[:60].replace('\n', ' ')
        print(f"      Text: {text_preview}...")
    
    # Topic classification breakdown  
    print(f"\n[TOPICS] TOPIC CLASSIFICATION:")
    topic_breakdown = {}
    for q in result.questions:
        topic = q.topic if hasattr(q, 'topic') else 'unknown'
        topic_breakdown[topic] = topic_breakdown.get(topic, 0) + 1
    
    unknown_count = topic_breakdown.get('unknown', 0)
    if total_questions > 0:
        print(f"   Unknown topics: {unknown_count} ({unknown_count/total_questions*100:.1f}%)")
    print(f"   Top 5 topics:")
    for topic, count in sorted(topic_breakdown.items(), key=lambda x: -x[1])[:5]:
        if topic != 'unknown':
            print(f"      {topic}: {count}")
    
    return {
        'name': name,
        'total_questions': total_questions,
        'non_verbal_questions': non_verbal_questions,
        'match_following': match_following,
        'questions_flagged_diagram': questions_flagged_diagram,
        'inferred_questions': inferred_questions,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence,
        'unknown_topics': unknown_count,
    }


if __name__ == '__main__':
    # Test on sample papers
    papers = [
        ('data/papers/AFCAT_2020_Memory.pdf', '2020 Memory-Based'),
        ('data/papers/AFCAT_2024_Aug10_Shift1_Memory.pdf', '2024 Aug 10 Shift 1'),
    ]
    
    results = []
    
    for pdf_path, name in papers:
        result = test_paper(pdf_path, name)
        if result:
            results.append(result)
    
    # Summary comparison
    if results:
        print(f"\n{'='*70}")
        print("[SUMMARY] COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        for r in results:
            figure_total = r['non_verbal_questions'] + r['match_following']
            figure_pct = figure_total / r['total_questions'] * 100 if r['total_questions'] > 0 else 0
            high_conf_pct = r['high_confidence'] / r['total_questions'] * 100 if r['total_questions'] > 0 else 0
            unknown_pct = r['unknown_topics'] / r['total_questions'] * 100 if r['total_questions'] > 0 else 0
            
            print(f"\n{r['name']}:")
            print(f"   Total: {r['total_questions']} questions")
            print(f"   Figure/Reasoning: {figure_total} ({figure_pct:.1f}%)")
            print(f"   High confidence: {r['high_confidence']} ({high_conf_pct:.1f}%)")
            print(f"   Unknown topics: {r['unknown_topics']} ({unknown_pct:.1f}%)")
        
        # Overall assessment
        total_qs = sum(r['total_questions'] for r in results)
        total_figures = sum(r['non_verbal_questions'] + r['match_following'] for r in results)
        total_high_conf = sum(r['high_confidence'] for r in results)
        total_unknown = sum(r['unknown_topics'] for r in results)
        
        if total_qs > 0:
            print(f"\n[OVERALL] AGGREGATE METRICS:")
            print(f"   Total questions: {total_qs}")
            print(f"   Figure detection rate: {total_figures/total_qs*100:.1f}%")
            print(f"   High confidence rate: {total_high_conf/total_qs*100:.1f}%")
            print(f"   Unknown topic rate: {total_unknown/total_qs*100:.1f}%")
            
            # Decision
            if total_figures/total_qs >= 0.15:
                print(f"\n   [OK] Figure detection is working (>15%)")
            else:
                print(f"\n   [WARN] Figure detection may be under-counting")
            
            if total_high_conf/total_qs >= 0.5:
                print(f"   [OK] Confidence levels are reasonable (>50% high)")
            else:
                print(f"   [WARN] Many low-confidence extractions need review")
            
            if total_unknown/total_qs <= 0.2:
                print(f"   [OK] Topic classification is reasonable (<20% unknown)")
            else:
                print(f"   [WARN] High unknown rate - classifier needs more keywords")
    
    print(f"\n{'='*70}\n")
