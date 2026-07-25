import json
import re
from pathlib import Path
import collections

DATA_DIR = Path("e:/afcatpyq3/data/processed")
Q_JSON_PATH = DATA_DIR / "Q.json"

def get_test_name(filename):
    if not filename:
        return "Unknown"
        
    # Match specific shifts: 2024_Aug09_Shift1
    m1 = re.search(r'(20\d\d_[A-Za-z]{3}\d\d(?:_Shift\d)?)', filename)
    if m1:
        raw = m1.group(1)
        # Convert "2024_Aug09_Shift1" to "AFCAT 2024 Aug 09 Shift 1"
        parts = raw.replace('_', ' ').split()
        return f"AFCAT {' '.join(parts)}"
        
    # Match date without shift: 2022_Feb13
    m2 = re.search(r'(20\d\d_[A-Za-z]{3}\d\d)', filename)
    if m2:
        raw = m2.group(1)
        parts = raw.replace('_', ' ').split()
        return f"AFCAT {' '.join(parts)}"
        
    # Match just the year
    m3 = re.search(r'(20\d\d)', filename)
    if m3:
        return f"AFCAT {m3.group(1)}"
        
    return "AFCAT Unknown"

def main():
    print("Unifying test names in Q.json...")
    with open(Q_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    test_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        
    for q in data:
        fn = q.get('file_name', '')
        test_name = get_test_name(fn)
        q['test_name'] = test_name
        
        sec = q.get('section')
        if sec:
            test_counts[test_name][sec] += 1
            
    with open(Q_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print("\nMerged Tests:")
    for tn in sorted(test_counts):
        print(f"{tn}: {dict(test_counts[tn])}")
        
    # Verify missing sections
    missing = 0
    for tn, counts in test_counts.items():
        if len(counts) < 4:
            print(f"[MISSING SECTIONS] {tn}: {dict(counts)}")
            missing += 1
            
    if missing == 0:
        print("\nSUCCESS! All tests now have 4 complete sections.")
    else:
        print(f"\nFound {missing} tests still missing sections. (Some older memory-based PDFs simply don't have all questions)")

if __name__ == "__main__":
    main()
