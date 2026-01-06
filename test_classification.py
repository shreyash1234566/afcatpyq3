"""
Test the new AFCAT_CLASSIFICATION system
"""
from models.question_classifier import (
    AFCATTopicClassifier, 
    AFCAT_CLASSIFICATION, 
    NON_VERBAL_PATTERNS
)

# Test 1: Check AFCAT_CLASSIFICATION loaded
print("=" * 60)
print("TEST 1: AFCAT_CLASSIFICATION Dictionary")
print("=" * 60)
print(f"AFCAT_CLASSIFICATION loaded with {len(AFCAT_CLASSIFICATION)} sections")
print(f"Sections: {list(AFCAT_CLASSIFICATION.keys())}")
print("Topics per section:")
for s in AFCAT_CLASSIFICATION:
    topics = AFCAT_CLASSIFICATION[s]["topics"]
    print(f"  {s}: {len(topics)} topics")
    # Show first 3 topic codes as sample
    sample = list(topics.keys())[:3]
    print(f"    Sample codes: {sample}")

print(f"\nNON_VERBAL_PATTERNS: {len(NON_VERBAL_PATTERNS)} patterns")

# Test 2: Initialize classifier
print("\n" + "=" * 60)
print("TEST 2: Classifier Initialization")
print("=" * 60)
clf = AFCATTopicClassifier()
print("Classifier initialized successfully")

# Test 3: Test classification with sample questions
print("\n" + "=" * 60)
print("TEST 3: Sample Question Classification")
print("=" * 60)

test_questions = [
    # Verbal Ability
    ("Choose the synonym of 'ABUNDANT'", 5, "verbal_ability", "VA_SYN"),
    ("Fill in the blank: The manager ___ the report yesterday.", 10, "verbal_ability", "VA_SEN_COMP"),
    ("Find the error in the sentence: She have completed the work.", 15, "verbal_ability", "VA_ERR"),
    
    # General Awareness
    ("Who was the first President of India?", 35, "general_awareness", "GA_POLITY"),
    ("Which river is known as the 'Sorrow of Bengal'?", 40, "general_awareness", "GA_GEO_IND"),
    ("The Dronacharya Award is given for excellence in:", 45, "general_awareness", "GA_AWARD"),
    
    # Reasoning
    ("If APPLE is coded as 50, then ORANGE is coded as:", 60, "reasoning", "RM_VR_CODE"),
    ("A is the father of B. B is the sister of C. How is A related to C?", 65, "reasoning", "RM_VR_BLOOD"),
    ("Which figure completes the pattern?", 70, "reasoning", "RM_NV_FIG"),
    
    # Numerical Ability
    ("A train 200m long crosses a platform in 15 seconds. Find its speed.", 85, "numerical_ability", "NA_TRAIN"),
    ("If the cost price is Rs. 500 and profit is 20%, find selling price.", 90, "numerical_ability", "NA_PL"),
    ("The average of 5 numbers is 20. Find the sum.", 95, "numerical_ability", "NA_AVG"),
]

print(f"Testing {len(test_questions)} sample questions...\n")

correct = 0
for question, q_num, expected_section, expected_topic in test_questions:
    result = clf.classify(question, question_number=q_num)
    
    section_match = result.section == expected_section
    topic_match = result.topic == expected_topic or expected_topic in result.topic
    
    status = "✓" if section_match else "✗"
    if section_match:
        correct += 1
    
    print(f"{status} Q{q_num}: {question[:50]}...")
    print(f"   Expected: {expected_section}/{expected_topic}")
    print(f"   Got:      {result.section}/{result.topic} (conf: {result.confidence:.2f})")
    print(f"   Method:   {result.method}")
    if result.subtopic:
        print(f"   Subtopic: {result.subtopic}")
    print()

print("=" * 60)
print(f"Section Accuracy: {correct}/{len(test_questions)} ({100*correct/len(test_questions):.1f}%)")
print("=" * 60)

# Test 4: Test utility functions
print("\n" + "=" * 60)
print("TEST 4: Utility Functions")
print("=" * 60)

# Test get_topic_label
print("get_topic_label tests:")
print(f"  VA_SYN -> {clf.get_topic_label('VA_SYN')}")
print(f"  NA_PER -> {clf.get_topic_label('NA_PER')}")
print(f"  RM_VR_CODE -> {clf.get_topic_label('RM_VR_CODE')}")
print(f"  GA_DEF -> {clf.get_topic_label('GA_DEF')}")

# Test get_section_from_code
print("\nget_section_from_code tests:")
print(f"  VA_SYN -> {clf.get_section_from_code('VA_SYN')}")
print(f"  GA_HIST_ANC -> {clf.get_section_from_code('GA_HIST_ANC')}")
print(f"  RM_NV_FIG -> {clf.get_section_from_code('RM_NV_FIG')}")
print(f"  NA_TW -> {clf.get_section_from_code('NA_TW')}")

# Test normalize_topic_name
print("\nnormalize_topic_name tests:")
print(f"  VA_IDIOM -> {clf.normalize_topic_name('VA_IDIOM')}")
print(f"  RM_VR_BLOOD -> {clf.normalize_topic_name('RM_VR_BLOOD')}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
