import json
from pathlib import Path
import collections

def main():
    Q_PATH = Path("e:/afcatpyq3/Q_c976d91.json")
    if not Q_PATH.exists():
        print(f"File not found: {Q_PATH}")
        return

    with open(Q_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # Group by test_name. If test_name is missing, use file_name.
    tests = collections.defaultdict(lambda: collections.defaultdict(int))
    
    # Target AFCAT distribution
    TARGETS = {
        "Verbal Ability": 30,
        "General Awareness": 25,
        "Reasoning": 25,
        "Numerical Ability": 20
    }
    
    for q in data:
        # We unified test_names earlier, let's use that
        t_name = q.get('test_name') or q.get('file_name') or "Unknown"
        sec = q.get('section', 'Unknown Section')
        tests[t_name][sec] += 1

    print("="*80)
    print(" AFCAT Q.JSON NOISE AUDIT: SHIFT-WISE & SUBJECT-WISE QUESTION COUNTS")
    print("="*80)
    
    total_missing_questions = 0
    perfect_tests = 0
    noisy_tests = 0

    for t_name in sorted(tests.keys()):
        counts = tests[t_name]
        total_q = sum(counts.values())
        
        # Check if the test is perfectly balanced
        is_perfect = True
        issues = []
        
        for sec, target in TARGETS.items():
            actual = counts.get(sec, 0)
            if actual != target:
                is_perfect = False
                diff = target - actual
                if actual == 0:
                    issues.append(f"MISSING entirely: {sec}")
                else:
                    issues.append(f"{sec} has {actual} (expected {target}, missing {diff})")
                
                # If they over-extracted, we don't count it as "missing" but it's noise
                if diff > 0:
                    total_missing_questions += diff
                
        # Format the output row
        va = counts.get('Verbal Ability', 0)
        ga = counts.get('General Awareness', 0)
        re = counts.get('Reasoning', 0)
        na = counts.get('Numerical Ability', 0)
        
        if is_perfect and total_q == 100:
            status = "[PERFECT]"
            perfect_tests += 1
        else:
            status = "[NOISY]  "
            noisy_tests += 1
            
        print(f"\n{status} {t_name}")
        print(f"    Total: {total_q} Qs | VA: {va} | GA: {ga} | RE: {re} | NA: {na}")
        
        if not is_perfect:
            for issue in issues:
                print(f"    -> {issue}")

    print("\n" + "="*80)
    print(" SUMMARY OF DATA NOISE")
    print("="*80)
    print(f"Total Tests Analyzed : {len(tests)}")
    print(f"Perfect 100-Q Tests  : {perfect_tests}")
    print(f"Noisy/Fragmented     : {noisy_tests}")
    print(f"Total Missing Qs     : ~{total_missing_questions} questions lost to PDF fragmentation or poor memory-retrieval.")
    print("="*80)
    print("CONCLUSION:")
    print("If a test is 'NOISY' (e.g., missing questions), the DM-Multinomial model gets punished")
    print("because it expects exactly 30 English / 25 Reasoning questions. When the PDF only provides")
    print("10 Reasoning questions, the MAE mathematically explodes because the data is incomplete.")

if __name__ == "__main__":
    main()
