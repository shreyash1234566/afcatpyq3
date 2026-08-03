import json
import re
from pathlib import Path
from collections import defaultdict

def extract_year(filename):
    match = re.search(r'(20\d\d)', filename or '')
    if match:
        return int(match.group(1))
    return None

def allocate_seats(total_seats, weights):
    """
    Allocate discrete integer seats (questions) proportionally using the Largest Remainder Method.
    `weights` is a dictionary of {item: float_weight}.
    """
    if total_seats == 0 or not weights:
        return {item: 0 for item in weights}
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        return {item: 0 for item in weights}
        
    quotients = {}
    seats = {}
    
    # Calculate exact fractional seats
    for item, weight in weights.items():
        fractional_seat = (weight / total_weight) * total_seats
        seats[item] = int(fractional_seat) # Base seat (floor)
        quotients[item] = fractional_seat - seats[item] # Remainder
        
    # Allocate remaining seats based on highest remainders
    remaining = total_seats - sum(seats.values())
    
    sorted_items = sorted(quotients.items(), key=lambda x: x[1], reverse=True)
    for i in range(int(remaining)):
        seats[sorted_items[i][0]] += 1
        
    return seats

def main():
    print("="*70)
    print(" 2026 SHIFT 2 MICRO TOPIC FORECASTER (WALK-FORWARD TESTING)")
    print("="*70)
    
    q_clean_path = Path("data/processed/Q_clean.json")
    if not q_clean_path.exists():
        print(f"ERROR: {q_clean_path} not found.")
        return
        
    with open(q_clean_path, 'r', encoding='utf-8') as f:
        qs = json.load(f)
        
    # 1. Calculate weighted historical distribution of Micro Topics
    # weights: 2025 = 2.0, 2024 = 1.0, older = 0.2
    topic_micro_weights = defaultdict(lambda: defaultdict(float))
    
    for q in qs:
        sec = q.get("section")
        top = q.get("topic")
        micro = q.get("micro_topic")
        year = extract_year(q.get("file_name"))
        
        if not all([sec, top, micro, year]):
            continue
            
        weight = 0.2
        if year == 2025:
            weight = 2.0
        elif year == 2024:
            weight = 1.0
            
        topic_micro_weights[top][micro] += weight
        
    # Normalize weights into probabilities
    topic_micro_probs = {}
    for top, micros in topic_micro_weights.items():
        total_w = sum(micros.values())
        topic_micro_probs[top] = {m: w/total_w for m, w in micros.items()}
        
    # 2. Load 2026 Predictions
    pred_path = Path("output/predictions_2026/afcat_2026_predictions.json")
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found.")
        return
        
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        
    ml_preds = pred_data.get("ml_predictions", [])
    if not ml_preds:
        print("ERROR: ml_predictions array not found or empty.")
        return
        
    # 3. Distribute predictions
    final_micro_predictions = []
    
    # Sort for report presentation
    section_order = {"Verbal Ability": 1, "General Awareness": 2, "Reasoning": 3, "Numerical Ability": 4}
    ml_preds.sort(key=lambda x: (section_order.get(x.get("Section", ""), 99), -x.get("Predicted_Questions", 0)))
    
    current_section = None
    
    for pred in ml_preds:
        sec = pred.get("Section")
        top = pred.get("Topic")
        top_name = pred.get("topic_name") or top
        pred_q = pred.get("Predicted_Questions", 0)
        
        if sec != current_section:
            print(f"\n" + "▬"*60)
            print(f" {sec.upper()} SECTION")
            print("▬"*60)
            current_section = sec
            
        if pred_q <= 0:
            continue
            
        probs = topic_micro_probs.get(top_name, {})
        if not probs:
            probs = topic_micro_probs.get(top, {}) # Try raw topic if topic_name wasn't found
            
        if not probs:
            print(f"  [!] Missing micro topics for: {top_name}. Defaulting to 'General {top_name}'")
            probs = {f"General {top_name}": 1.0}
            
        # Allocate exactly 'pred_q' seats
        allocated = allocate_seats(pred_q, probs)
        
        print(f"\n   {top_name} (Predicted: {pred_q} Qs)")
        
        # Sort allocated descending
        allocated_sorted = sorted(allocated.items(), key=lambda x: x[1], reverse=True)
        for m_topic, count in allocated_sorted:
            if count > 0:
                print(f"       ├── {count} Qs : {m_topic}")
                final_micro_predictions.append({
                    "Section": sec,
                    "Broad_Topic": top_name,
                    "Micro_Topic": m_topic,
                    "Predicted_Questions": count,
                    "Probability_Weight": round(probs.get(m_topic, 0), 4)
                })
                
    # 4. Save results
    out_path = Path("output/predictions_2026/afcat_2026_micro_predictions.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_micro_predictions, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*70)
    print(f"✅ Micro topic predictions successfully saved to:\n   {out_path}")
    print("="*70)

if __name__ == "__main__":
    main()
