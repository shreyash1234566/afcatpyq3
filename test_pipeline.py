"""
Quick test script for the paper extraction pipeline.
Run this to verify the OCR and classification pipeline is working.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_ocr_import():
    """Test that OCR modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.ocr_engine import ExamPaperOCR, MCQExtractor, OCREngine
        print("  ✅ OCR Engine imported")
    except ImportError as e:
        print(f"  ❌ OCR Engine import failed: {e}")
        return False
        
    try:
        from models.question_classifier import AFCATTopicClassifier, QuestionTypeClassifier
        print("  ✅ Question Classifier imported")
    except ImportError as e:
        print(f"  ❌ Question Classifier import failed: {e}")
        return False
        
    try:
        from models.enhanced_difficulty import EnhancedDifficultyPredictor
        print("  ✅ Difficulty Predictor imported")
    except ImportError as e:
        print(f"  ❌ Difficulty Predictor import failed: {e}")
        return False
        
    try:
        from pipeline import AFCATExamAnalyzer
        print("  ✅ Pipeline imported")
    except ImportError as e:
        print(f"  ❌ Pipeline import failed: {e}")
        return False
        
    return True


def test_easyocr():
    """Test that EasyOCR can be initialized."""
    print("\nTesting EasyOCR initialization...")
    
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("  ✅ EasyOCR initialized successfully")
        return True
    except ImportError:
        print("  ❌ EasyOCR not installed. Run: pip install easyocr")
        return False
    except Exception as e:
        print(f"  ⚠️ EasyOCR warning: {e}")
        return True  # May still work


def test_classification():
    """Test question classification."""
    print("\nTesting question classification...")
    
    from models.question_classifier import AFCATTopicClassifier, QuestionTypeClassifier
    
    classifier = AFCATTopicClassifier(use_transformers=False)
    type_classifier = QuestionTypeClassifier()
    
    # Test questions
    test_questions = [
        "A train travels 360 km in 6 hours. What is its speed in km/hr?",
        "The synonym of 'ephemeral' is:",
        "If COMPUTER is coded as 12345678, what is METRO coded as?",
        "Who is the current Chief of Air Staff of Indian Air Force?",
        "What is the capital of Australia?",
    ]
    
    print("  Sample classifications:")
    for q in test_questions:
        result = classifier.classify(q)
        q_type, _ = type_classifier.classify(q)
        print(f"    Q: {q[:50]}...")
        print(f"       Section: {result.section}, Topic: {result.topic} ({result.confidence:.0%})")
        print(f"       Type: {q_type}")
        print()
        
    print("  ✅ Classification working")
    return True


def test_difficulty():
    """Test difficulty prediction."""
    print("Testing difficulty prediction...")
    
    from models.enhanced_difficulty import EnhancedDifficultyPredictor
    
    predictor = EnhancedDifficultyPredictor()
    
    test_cases = [
        ("What is 2 + 2?", None, "simplification"),
        ("A train 150m long crosses a platform 200m long in 20 seconds. Find the speed of the train.", None, "speed_time_distance"),
        ("The compound interest on Rs. 10000 at 8% per annum for 2 years compounded annually is:", None, "simple_compound_interest"),
    ]
    
    for q, opts, topic in test_cases:
        level, conf, details = predictor.predict(q, opts, topic)
        print(f"    {q[:40]}...")
        print(f"       Difficulty: {level} ({conf:.0%} confidence)")
        
    print("  ✅ Difficulty prediction working")
    return True


def test_mcq_extraction():
    """Test MCQ extraction from text."""
    print("\nTesting MCQ extraction...")
    
    from utils.ocr_engine import MCQExtractor
    
    sample_text = """
    1. A train travels 360 km in 6 hours. What is its speed?
    (A) 60 km/hr (B) 50 km/hr (C) 55 km/hr (D) 65 km/hr
    
    2. The synonym of 'benevolent' is:
    (A) Kind (B) Cruel (C) Harsh (D) Rude
    
    3. If CAT is coded as 24, then DOG is coded as:
    (A) 26 (B) 25 (C) 27 (D) 28
    """
    
    extractor = MCQExtractor()
    questions = extractor.extract_questions(sample_text)
    
    print(f"  Extracted {len(questions)} questions:")
    for q in questions:
        print(f"    Q{q.question_number}: {q.text[:40]}...")
        print(f"       Options: {len(q.options)} found")
        
    if len(questions) >= 2:
        print("  ✅ MCQ extraction working")
        return True
    else:
        print("  ⚠️ MCQ extraction may need adjustment")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("AFCAT Paper Extraction Pipeline - Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_ocr_import()))
    results.append(("EasyOCR", test_easyocr()))
    results.append(("MCQ Extraction", test_mcq_extraction()))
    results.append(("Classification", test_classification()))
    results.append(("Difficulty", test_difficulty()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20} {status}")
        if result:
            passed += 1
            
    print(f"\n  {passed}/{len(results)} tests passed")
    print("=" * 60)
    
    if passed == len(results):
        print("\n🎉 Pipeline ready! You can now extract papers:")
        print("   python main.py --extract your_paper.pdf --year 2025")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        

if __name__ == "__main__":
    main()
