"""
AFCAT 2026 MODEL VALIDATION & FIX SYSTEM
=========================================
1. Analyzes current model accuracy
2. Identifies data quality issues
3. Fixes prediction logic with realistic constraints
4. Validates against historical patterns
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("output/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# STEP 1: DATA QUALITY ANALYSIS
# ==========================================

def analyze_data_quality(data_file):
    """Deep dive into data quality issues"""
    print("\n" + "="*70)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*70)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Analysis metrics
    total_questions = len(raw_data)
    years_found = defaultdict(int)
    topics_found = defaultdict(int)
    sections_found = defaultdict(int)
    missing_data = {'no_topic': 0, 'no_year': 0, 'no_section': 0, 'no_text': 0}
    
    for q in raw_data:
        # Year analysis
        year = q.get('year') or q.get('Year')
        if year:
            years_found[int(year)] += 1
        else:
            missing_data['no_year'] += 1
        
        # Topic analysis
        topic = q.get('topic_name') or q.get('topic') or q.get('Topic')
        if topic:
            topics_found[topic] += 1
        else:
            missing_data['no_topic'] += 1
        
        # Section analysis
        section = q.get('subject') or q.get('section') or q.get('Subject')
        if section:
            sections_found[section] += 1
        else:
            missing_data['no_section'] += 1
        
        # Text quality
        text = q.get('question_text') or q.get('question') or ''
        if not text or len(text) < 10:
            missing_data['no_text'] += 1
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total Questions: {total_questions}")
    print(f"   Unique Years: {len(years_found)}")
    print(f"   Unique Topics: {len(topics_found)}")
    print(f"   Unique Sections: {len(sections_found)}")
    
    print(f"\n⚠️  DATA COMPLETENESS:")
    print(f"   Missing Year: {missing_data['no_year']} ({missing_data['no_year']/total_questions*100:.1f}%)")
    print(f"   Missing Topic: {missing_data['no_topic']} ({missing_data['no_topic']/total_questions*100:.1f}%)")
    print(f"   Missing Section: {missing_data['no_section']} ({missing_data['no_section']/total_questions*100:.1f}%)")
    print(f"   Invalid Text: {missing_data['no_text']} ({missing_data['no_text']/total_questions*100:.1f}%)")
    
    print(f"\n📅 YEAR-WISE DISTRIBUTION:")
    for year in sorted(years_found.keys()):
        print(f"   {year}: {years_found[year]} questions")
    
    print(f"\n🎯 TOP 20 TOPICS BY FREQUENCY:")
    sorted_topics = sorted(topics_found.items(), key=lambda x: x[1], reverse=True)
    for topic, count in sorted_topics[:20]:
        print(f"   {topic:<40}: {count:>4} questions")
    
    # Detect anomalies
    print(f"\n🚨 ANOMALY DETECTION:")
    avg_questions_per_topic = total_questions / len(topics_found)
    anomalies = []
    for topic, count in sorted_topics:
        if count > avg_questions_per_topic * 3:
            anomalies.append((topic, count))
    
    if anomalies:
        print(f"   Topics with unusually high frequency (>3x average):")
        for topic, count in anomalies:
            print(f"   ⚠️  {topic}: {count} ({count/avg_questions_per_topic:.1f}x average)")
    else:
        print("   ✓ No major anomalies detected")
    
    # Section balance
    print(f"\n⚖️  SECTION BALANCE:")
    total_with_section = sum(sections_found.values())
    for section, count in sections_found.items():
        pct = count / total_with_section * 100
        print(f"   {section:<30}: {count:>4} ({pct:>5.1f}%)")
    
    return {
        'total_questions': total_questions,
        'years': dict(years_found),
        'topics': dict(topics_found),
        'sections': dict(sections_found),
        'missing_data': missing_data,
        'avg_questions_per_topic': avg_questions_per_topic,
        'anomalies': anomalies
    }

# ==========================================
# STEP 2: HISTORICAL PATTERN ANALYSIS
# ==========================================

def analyze_historical_patterns(data_file):
    """Analyze actual AFCAT patterns from historical data"""
    print("\n" + "="*70)
    print("📖 HISTORICAL PATTERN ANALYSIS")
    print("="*70)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Group by year and topic
    year_topic_counts = defaultdict(lambda: defaultdict(int))
    year_section_counts = defaultdict(lambda: defaultdict(int))
    
    for q in raw_data:
        year = q.get('year') or q.get('Year')
        topic = q.get('topic_name') or q.get('topic')
        section = q.get('subject') or q.get('section')
        
        if year and topic:
            year_topic_counts[int(year)][topic] += 1
        if year and section:
            year_section_counts[int(year)][section] += 1
    
    # Calculate max questions per topic per year
    max_per_topic = {}
    avg_per_topic = {}
    
    for year, topics in year_topic_counts.items():
        for topic, count in topics.items():
            if topic not in max_per_topic:
                max_per_topic[topic] = count
                avg_per_topic[topic] = []
            else:
                max_per_topic[topic] = max(max_per_topic[topic], count)
            avg_per_topic[topic].append(count)
    
    # Calculate stats
    for topic in avg_per_topic:
        avg_per_topic[topic] = np.mean(avg_per_topic[topic])
    
    print(f"\n📊 REALISTIC TOPIC LIMITS (Based on Historical Data):")
    print(f"{'TOPIC':<40} | {'MAX':<6} | {'AVG':<6} | {'YEARS':<6}")
    print("-" * 70)
    
    sorted_topics = sorted(max_per_topic.items(), key=lambda x: x[1], reverse=True)
    for topic, max_count in sorted_topics[:30]:
        avg = avg_per_topic[topic]
        years = len([1 for y, topics in year_topic_counts.items() if topic in topics])
        print(f"{topic:<40} | {max_count:<6} | {avg:<6.1f} | {years:<6}")
    
    # Overall statistics
    all_max_values = list(max_per_topic.values())
    
    if not all_max_values:
        print(f"\n❌ NO HISTORICAL DATA AVAILABLE")
        print(f"   Reason: All questions are missing year information")
        print(f"   Impact: Cannot calculate historical patterns")
        print(f"\n💡 SOLUTION: Run data repair script first:")
        print(f"      python scripts/repair_year_data.py")
        return {
            'max_per_topic': {},
            'avg_per_topic': {},
            'overall_max': 6,  # Safe default
            'percentile_90': 5,
            'percentile_95': 6,
            'section_stats': {}
        }
    
    print(f"\n📈 OVERALL LIMITS:")
    print(f"   Absolute Maximum: {max(all_max_values)} questions/topic/year")
    print(f"   Average Maximum: {np.mean(all_max_values):.1f} questions/topic/year")
    print(f"   90th Percentile: {np.percentile(all_max_values, 90):.1f}")
    print(f"   95th Percentile: {np.percentile(all_max_values, 95):.1f}")
    
    # Section-wise totals
    print(f"\n📋 SECTION-WISE TOTALS (Per Year):")
    section_stats = defaultdict(list)
    for year, sections in year_section_counts.items():
        for section, count in sections.items():
            section_stats[section].append(count)
    
    for section, counts in section_stats.items():
        print(f"   {section:<30}: {np.mean(counts):.1f} ± {np.std(counts):.1f} (max: {max(counts)})")
    
    return {
        'max_per_topic': max_per_topic,
        'avg_per_topic': avg_per_topic,
        'overall_max': max(all_max_values),
        'percentile_90': np.percentile(all_max_values, 90),
        'percentile_95': np.percentile(all_max_values, 95),
        'section_stats': {k: {'mean': np.mean(v), 'std': np.std(v), 'max': max(v)} 
                          for k, v in section_stats.items()}
    }

# ==========================================
# STEP 3: MODEL VALIDATION
# ==========================================

def validate_current_model(predictions_file):
    """Validate predictions against realistic bounds"""
    print("\n" + "="*70)
    print("🔍 MODEL VALIDATION")
    print("="*70)
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data.get('ml_predictions', [])
    
    print(f"\n⚠️  UNREALISTIC PREDICTIONS (>6 questions):")
    print(f"{'TOPIC':<40} | {'PREDICTED':<10} | {'SECTION'}")
    print("-" * 70)
    
    unrealistic = []
    for pred in predictions:
        if pred['predicted_count'] > 6:
            unrealistic.append(pred)
            print(f"{pred['topic_name']:<40} | {pred['predicted_count']:<10.1f} | {pred['section']}")
    
    print(f"\n📊 PREDICTION STATISTICS:")
    all_preds = [p['predicted_count'] for p in predictions]
    print(f"   Mean Prediction: {np.mean(all_preds):.2f}")
    print(f"   Median Prediction: {np.median(all_preds):.2f}")
    print(f"   Max Prediction: {max(all_preds):.2f}")
    print(f"   Std Deviation: {np.std(all_preds):.2f}")
    print(f"   Topics >6 questions: {len(unrealistic)}")
    print(f"   Topics >8 questions: {len([p for p in predictions if p['predicted_count'] > 8])}")
    
    # Section-wise totals
    section_totals = defaultdict(float)
    for pred in predictions:
        section_totals[pred['section']] += pred['predicted_count']
    
    print(f"\n📋 PREDICTED SECTION TOTALS:")
    for section, total in section_totals.items():
        print(f"   {section:<30}: {total:.1f} questions")
    
    return {
        'unrealistic_count': len(unrealistic),
        'unrealistic_predictions': unrealistic,
        'mean_prediction': np.mean(all_preds),
        'max_prediction': max(all_preds),
        'section_totals': dict(section_totals)
    }

# ==========================================
# STEP 4: FIXED MODEL WITH CONSTRAINTS
# ==========================================

def train_constrained_model(data_file, historical_limits):
    """Train model with realistic constraints"""
    print("\n" + "="*70)
    print("🔧 TRAINING CONSTRAINED MODEL")
    print("="*70)
    
    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Build topic data
    topic_data = defaultdict(lambda: {'years': [], 'count': 0})
    for q in raw_data:
        topic = q.get('topic_name') or q.get('topic')
        year = q.get('year') or q.get('Year')
        if topic and year:
            topic_data[topic]['years'].append(int(year))
            topic_data[topic]['count'] += 1
    
    # Extract features
    X_list, y_list, topics = [], [], []
    for topic, data in topic_data.items():
        if data['count'] < 3:  # Need minimum data
            continue
        
        years = sorted(set(data['years']))
        year_counts = [data['years'].count(y) for y in years]
        
        if len(year_counts) < 2:
            continue
        
        # Features: avg, recent_avg, trend, recency, max, min
        avg = np.mean(year_counts)
        recent_avg = np.mean(year_counts[-2:]) if len(year_counts) >= 2 else avg
        trend = np.polyfit(range(len(year_counts)), year_counts, 1)[0] if len(year_counts) >= 3 else 0
        recency = 2026 - max(years)
        max_count = max(year_counts)
        min_count = min(year_counts)
        
        X_list.append([avg, recent_avg, trend, recency, max_count, min_count])
        y_list.append(avg)  # Target is average per year
        topics.append(topic)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n📊 Training Data:")
    print(f"   Topics: {len(topics)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target Range: {y.min():.2f} - {y.max():.2f}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ensemble
    models = {}
    predictions = {}
    
    if XGBOOST_AVAILABLE:
        print("\n🌲 Training XGBoost...")
        xgb = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, 
                          random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"   MAE: {mae:.3f} | R²: {r2:.3f}")
        models['xgb'] = xgb
    
    if LIGHTGBM_AVAILABLE:
        print("\n🌲 Training LightGBM...")
        lgb = LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                           random_state=42, verbose=-1)
        lgb.fit(X_train, y_train)
        y_pred = lgb.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"   MAE: {mae:.3f} | R²: {r2:.3f}")
        models['lgb'] = lgb
    
    if SKLEARN_AVAILABLE:
        print("\n🌲 Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=50, max_depth=6, 
                                  random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"   MAE: {mae:.3f} | R²: {r2:.3f}")
        models['rf'] = rf
    
    # Ensemble prediction with constraints
    print("\n🎯 Generating Constrained Predictions...")
    ensemble_preds = np.zeros(len(X))
    for model_name, model in models.items():
        ensemble_preds += model.predict(X) / len(models)
    
    # Apply realistic constraints
    constrained_predictions = []
    for i, topic in enumerate(topics):
        raw_pred = max(0, ensemble_preds[i])
        
        # Apply topic-specific historical limit
        if topic in historical_limits['max_per_topic']:
            historical_max = historical_limits['max_per_topic'][topic]
            historical_avg = historical_limits['avg_per_topic'][topic]
            # Cap at 1.2x historical max or historical_max + 2, whichever is lower
            cap = min(historical_max * 1.2, historical_max + 2)
        else:
            # For unseen topics, use percentile limits
            cap = historical_limits['percentile_90']
        
        # Absolute maximum (AFCAT rarely asks >6 from one narrow topic)
        absolute_max = 6.0
        
        # Special cases
        if any(word in topic.lower() for word in ['comprehension', 'reading']):
            absolute_max = 8.0  # RC can be higher
        elif any(word in topic.lower() for word in ['cloze', 'passage']):
            absolute_max = 7.0
        
        final_pred = min(raw_pred, cap, absolute_max)
        
        constrained_predictions.append({
            'topic': topic,
            'raw_prediction': round(raw_pred, 2),
            'constrained_prediction': round(final_pred, 1),
            'historical_max': historical_limits['max_per_topic'].get(topic, 'N/A'),
            'applied_cap': round(min(cap, absolute_max), 1)
        })
    
    print(f"   ✓ Generated {len(constrained_predictions)} predictions")
    
    return constrained_predictions, models

# ==========================================
# STEP 5: ACCURACY MATRIX
# ==========================================

def generate_accuracy_matrix(original_preds, constrained_preds, historical_limits):
    """Generate comprehensive accuracy comparison"""
    print("\n" + "="*70)
    print("📊 ACCURACY MATRIX")
    print("="*70)
    
    # Compare predictions
    comparison = []
    for orig in original_preds:
        topic = orig['topic_name']
        orig_count = orig['predicted_count']
        
        # Find constrained version
        const = next((p for p in constrained_preds if p['topic'] == topic), None)
        const_count = const['constrained_prediction'] if const else orig_count
        
        hist_max = historical_limits['max_per_topic'].get(topic, 'N/A')
        hist_avg = historical_limits['avg_per_topic'].get(topic, 'N/A')
        
        comparison.append({
            'topic': topic,
            'original_pred': orig_count,
            'constrained_pred': const_count,
            'historical_max': hist_max if hist_max != 'N/A' else 0,
            'historical_avg': hist_avg if hist_avg != 'N/A' else 0,
            'improvement': orig_count - const_count
        })
    
    # Print comparison
    print(f"\n🔄 PREDICTION COMPARISON:")
    print(f"{'TOPIC':<35} | {'ORIGINAL':<9} | {'FIXED':<6} | {'HIST MAX':<9} | {'DIFF':<6}")
    print("-" * 80)
    
    for c in sorted(comparison, key=lambda x: x['improvement'], reverse=True)[:20]:
        if c['improvement'] > 0:
            print(f"{c['topic']:<35} | {c['original_pred']:<9.1f} | {c['constrained_pred']:<6.1f} | "
                  f"{c['historical_max']:<9} | {c['improvement']:<6.1f}")
    
    # Statistics
    orig_unrealistic = len([c for c in comparison if c['original_pred'] > 6])
    fixed_unrealistic = len([c for c in comparison if c['constrained_pred'] > 6])
    
    print(f"\n📈 IMPROVEMENT METRICS:")
    print(f"   Original Unrealistic (>6): {orig_unrealistic}")
    print(f"   Fixed Unrealistic (>6): {fixed_unrealistic}")
    print(f"   Improvement: {orig_unrealistic - fixed_unrealistic} topics fixed")
    print(f"   Average Original: {np.mean([c['original_pred'] for c in comparison]):.2f}")
    print(f"   Average Fixed: {np.mean([c['constrained_pred'] for c in comparison]):.2f}")
    print(f"   Max Original: {max([c['original_pred'] for c in comparison]):.2f}")
    print(f"   Max Fixed: {max([c['constrained_pred'] for c in comparison]):.2f}")
    
    return comparison

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("\n" + "="*70)
    print("🎯 AFCAT MODEL VALIDATION & FIX SYSTEM")
    print("="*70)
    
    data_file = DATA_DIR / "Q_normalized.json"
    pred_file = Path("output/predictions_2026/afcat_2026_predictions.json")
    
    # Step 1: Data Quality Analysis
    quality_report = analyze_data_quality(data_file)
    
    # Step 2: Historical Pattern Analysis
    historical_limits = analyze_historical_patterns(data_file)
    
    # Step 3: Validate Current Model
    if pred_file.exists():
        validation_report = validate_current_model(pred_file)
        
        # Load original predictions for comparison
        with open(pred_file, 'r') as f:
            original_data = json.load(f)
        original_preds = original_data.get('ml_predictions', [])
    else:
        print(f"\n⚠️  Prediction file not found: {pred_file}")
        original_preds = []
    
    # Step 4: Train Constrained Model
    constrained_preds, models = train_constrained_model(data_file, historical_limits)
    
    # Step 5: Generate Accuracy Matrix
    if original_preds:
        comparison = generate_accuracy_matrix(original_preds, constrained_preds, historical_limits)
    
    # Save reports
    report = {
        'data_quality': quality_report,
        'historical_limits': {
            'overall_max': historical_limits['overall_max'],
            'percentile_90': historical_limits['percentile_90'],
            'percentile_95': historical_limits['percentile_95'],
        },
        'constrained_predictions': constrained_preds,
        'recommendations': {
            'max_per_topic': 6,
            'max_for_rc': 8,
            'max_for_cloze': 7,
            'use_historical_caps': True,
            'apply_percentile_limits': True
        }
    }
    
    output_file = OUTPUT_DIR / "validation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Report saved: {output_file}")
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70)
    print("\n📋 KEY FINDINGS:")
    print(f"   1. Data completeness: {100 - quality_report['missing_data']['no_topic']/quality_report['total_questions']*100:.1f}%")
    print(f"   2. Historical max per topic: {historical_limits['overall_max']}")
    print(f"   3. Recommended cap: {historical_limits['percentile_90']:.0f} questions")
    if original_preds:
        print(f"   4. Topics fixed: {len([c for c in comparison if c['improvement'] > 0])}")
    print("\n🔧 NEXT STEPS:")
    print("   → Use constrained_predictions from validation_report.json")
    print("   → Update main model with realistic caps")
    print("   → Re-run predictions with fixed logic")

if __name__ == "__main__":
    main()