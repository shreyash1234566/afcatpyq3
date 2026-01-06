#!/usr/bin/env python3
"""
Compare API classification results against ground truth.
"""
import json
from collections import Counter

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_results():
    # Load ground truth
    ground_truth = load_json("data/ground_truth/AFCAT_2017_Memory_ground_truth.json")
    gt_questions = ground_truth["questions"]
    
    # Load API results
    api_results = load_json("output/api_classification/afcat_2017_1_api_classified.json")
    api_questions = api_results["questions"]
    
    # Build lookup by question number
    gt_by_num = {q["question_number"]: q for q in gt_questions}
    api_by_num = {q["question_number"]: q for q in api_questions}
    
    # Compare sections
    print("=" * 70)
    print("GROUND TRUTH vs API CLASSIFICATION COMPARISON")
    print("=" * 70)
    
    # Section distribution comparison
    print("\n📊 Section Distribution:")
    print("-" * 50)
    print(f"{'Section':<25} {'Ground Truth':>12} {'API Result':>12}")
    print("-" * 50)
    
    gt_sections = Counter(q["section"] for q in gt_questions)
    api_sections = Counter(q["section"] for q in api_questions)
    
    for section in ["verbal_ability", "general_awareness", "numerical_ability", "reasoning"]:
        gt_count = gt_sections.get(section, 0)
        api_count = api_sections.get(section, 0)
        diff = "✅" if gt_count == api_count else f"❌ ({api_count - gt_count:+d})"
        print(f"{section:<25} {gt_count:>12} {api_count:>12}  {diff}")
    
    if "unknown" in api_sections:
        print(f"{'unknown':<25} {'0':>12} {api_sections['unknown']:>12}  ⚠️")
    
    # Per-question section accuracy
    print("\n📈 Per-Question Section Accuracy:")
    print("-" * 50)
    
    correct = 0
    wrong = []
    unknown_count = 0
    
    for qnum in range(1, 101):
        gt_q = gt_by_num.get(qnum)
        api_q = api_by_num.get(qnum)
        
        if not gt_q or not api_q:
            continue
        
        gt_section = gt_q["section"]
        api_section = api_q["section"]
        
        if api_section == "unknown":
            unknown_count += 1
            wrong.append((qnum, gt_section, api_section))
        elif gt_section == api_section:
            correct += 1
        else:
            wrong.append((qnum, gt_section, api_section))
    
    total = len(gt_questions)
    accuracy = (correct / total) * 100
    
    print(f"✅ Correct:   {correct}/{total} ({accuracy:.1f}%)")
    print(f"❌ Wrong:     {len(wrong) - unknown_count}/{total} ({((len(wrong)-unknown_count)/total)*100:.1f}%)")
    print(f"⚠️  Unknown:   {unknown_count}/{total} ({(unknown_count/total)*100:.1f}%)")
    
    # Show misclassified questions
    if wrong:
        print("\n🔍 Misclassified Questions:")
        print("-" * 50)
        print(f"{'Q#':>4}  {'Ground Truth':<20} {'API Result':<20}")
        print("-" * 50)
        for qnum, gt_sec, api_sec in wrong[:20]:  # Show first 20
            print(f"{qnum:>4}  {gt_sec:<20} {api_sec:<20}")
        if len(wrong) > 20:
            print(f"  ... and {len(wrong) - 20} more")
    
    # Analysis of systematic errors
    print("\n📋 Error Pattern Analysis:")
    print("-" * 50)
    
    error_patterns = Counter()
    for qnum, gt_sec, api_sec in wrong:
        if api_sec != "unknown":
            pattern = f"{gt_sec} → {api_sec}"
            error_patterns[pattern] += 1
    
    for pattern, count in error_patterns.most_common(10):
        print(f"  {pattern}: {count} questions")
    
    print("\n" + "=" * 70)
    print(f"🎯 OVERALL SECTION ACCURACY: {accuracy:.1f}%")
    print("=" * 70)
    
    return accuracy

if __name__ == "__main__":
    compare_results()
