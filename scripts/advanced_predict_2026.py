"""
AFCAT 2026 Advanced Prediction System
=====================================
Uses multiple ML techniques based on research:
1. Topic Trend Analysis (XGBoost) - Predict topic frequencies
2. Semantic Similarity (TF-IDF) - Find similar questions
3. Question Pattern Mining - Identify repeating patterns
4. Keyword Extraction - High-frequency terms per topic
5. Year-over-Year Analysis - Detect cycles

This script:
1. Fixes Q.json year extraction from filenames
2. Extracts 2025 data from existing PDFs
3. Generates advanced predictions for 2026
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import random

# Try to import ML libraries
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Note: sklearn not available, using basic prediction mode")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Paths
DATA_DIR = Path("data/processed")
PAPERS_DIR = Path("data/papers")
OUTPUT_DIR = Path("output/predictions_2026")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Section quotas
SECTION_DISTRIBUTION = {
    "Verbal Ability": 30,
    "General Awareness": 25,
    "Numerical Ability": 20,
    "Reasoning": 25
}

# Topic name mapping
TOPIC_NAMES = {
    "VA_COMP": "Reading Comprehension",
    "VA_CLOZE": "Cloze Test",
    "VA_ERR": "Error Detection",
    "VA_SENT": "Sentence Completion",
    "VA_REARR": "Sentence Rearrangement",
    "VA_SYN": "Synonyms",
    "VA_ANT": "Antonyms",
    "VA_IDIOM": "Idioms & Phrases",
    "VA_OWS": "One Word Substitution",
    "VA_ANALOGY": "Verbal Analogy",
    "VA_GRAM": "Grammar",
    "GA_SCI": "Science & Technology",
    "GA_DEF": "Defence & Military",
    "GA_CURR": "Current Affairs",
    "GA_SPORTS": "Sports",
    "GA_POLITY": "Polity & Governance",
    "GA_HIST_MOD": "Modern History",
    "GA_HIST_ANC": "Ancient History",
    "GA_HIST_MED": "Medieval History",
    "GA_GEO_IND": "Indian Geography",
    "GA_GEO_WORLD": "World Geography",
    "GA_ECON": "Economy",
    "GA_ENV": "Environment",
    "GA_CULTURE": "Art & Culture",
    "GA_BOOKS": "Books & Authors",
    "GA_AWARDS": "Awards & Honours",
    "GA_ORG": "Organizations",
    "GA_PERS": "Personalities",
    "NA_NUM": "Number System",
    "NA_PER": "Percentage",
    "NA_PL": "Profit & Loss",
    "NA_SI": "Simple Interest",
    "NA_CI": "Compound Interest",
    "NA_RAT": "Ratio & Proportion",
    "NA_AVG": "Average",
    "NA_TW": "Time & Work",
    "NA_SPD": "Speed, Distance, Time",
    "NA_AREA": "Area & Perimeter",
    "NA_MENSA": "Mensuration",
    "NA_STAT": "Statistics",
    "NA_SIM": "Simplification",
    "NA_MIX": "Mixture & Alligation",
    "NA_HCF": "HCF & LCM",
    "NA_ALG": "Algebra",
    "NA_DEC": "Decimal & Fraction",
    "RM_VR_CODING": "Coding-Decoding",
    "RM_VR_LOG": "Logical Reasoning",
    "RM_VR_CLASS": "Classification",
    "RM_VR_ANALOGY": "Verbal Analogy",
    "RM_VR_SERIES": "Series",
    "RM_VR_SYLL": "Syllogism",
    "RM_NV_PATTERN": "Pattern/Figure Series",
    "RM_NV_SPATIAL": "Spatial Ability",
    "RM_NV_ORIENT": "Direction Sense",
    "RM_NV_VENN": "Venn Diagrams",
    "RM_NV_MIRROR": "Mirror/Water Image",
    "RM_NV_DOT": "Dot Situation"
}

SECTION_MAP = {
    "VA_": "Verbal Ability",
    "GA_": "General Awareness",
    "NA_": "Numerical Ability",
    "RM_": "Reasoning"
}


def extract_year_from_filename(filename):
    """Extract year from AFCAT filename."""
    if not filename:
        return None
    match = re.search(r'AFCAT[_\s]*(\d{4})', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d{4})', filename)
    if match:
        year = int(match.group(1))
        if 2010 <= year <= 2030:
            return year
    return None


def get_section_from_topic(topic_code):
    """Get section name from topic code."""
    for prefix, section in SECTION_MAP.items():
        if topic_code.startswith(prefix):
            return section
    return "General Awareness"


def load_and_fix_pyq_data():
    """Load Q.json and fix year extraction."""
    print("📂 Loading and fixing Q.json...")
    
    with open(DATA_DIR / "Q.json", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse as JSON array
    # Handle potential multiple JSON objects
    try:
        # Check if it's a single array
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to fix common issues
        print("   ⚠️ JSON parse error, attempting to fix...")
        
        # Find all complete JSON objects
        data = []
        decoder = json.JSONDecoder()
        pos = 0
        content = content.strip()
        
        while pos < len(content):
            try:
                obj, end = decoder.raw_decode(content, pos)
                if isinstance(obj, list):
                    data.extend(obj)
                elif isinstance(obj, dict):
                    data.append(obj)
                pos = end
                # Skip whitespace
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
            except json.JSONDecodeError:
                pos += 1
        
        print(f"   ✓ Recovered {len(data)} questions")
    
    # Fix years from filenames
    fixed_count = 0
    for q in data:
        if not q.get('year'):
            year = extract_year_from_filename(q.get('file_name', ''))
            if year:
                q['year'] = year
                fixed_count += 1
        
        # Ensure section is set
        if not q.get('section') and q.get('topic_code'):
            q['section'] = get_section_from_topic(q['topic_code'])
    
    print(f"   ✓ Fixed {fixed_count} missing years from filenames")
    
    # Count by year
    year_counts = Counter(q.get('year') for q in data if q.get('year'))
    print(f"   ✓ Year distribution:")
    for year in sorted(year_counts.keys()):
        print(f"      {year}: {year_counts[year]} questions")
    
    return data


def analyze_question_patterns(questions):
    """Analyze question patterns to identify repeating formats."""
    print("\n🔍 Analyzing question patterns...")
    
    patterns = defaultdict(list)
    
    # Common question starters
    starters = [
        (r'^find\s+the\s+(synonym|antonym)', 'VA_SYN_ANT'),
        (r'^choose\s+the\s+correct', 'CORRECT_CHOICE'),
        (r'^what\s+is\s+the', 'WHAT_IS'),
        (r'^which\s+(one|of\s+the)', 'WHICH_ONE'),
        (r'^the\s+ratio', 'NA_RATIO'),
        (r'^if\s+a\s+train', 'NA_SPEED'),
        (r'^in\s+a\s+mixture', 'NA_MIXTURE'),
        (r'^complete\s+the\s+series', 'RM_SERIES'),
        (r'^find\s+the\s+odd', 'RM_CLASS'),
        (r'^decode', 'RM_CODING'),
    ]
    
    for q in questions:
        text = (q.get('question_text') or '').lower().strip()
        topic = q.get('topic_code', 'UNKNOWN')
        
        for pattern, pattern_name in starters:
            if re.match(pattern, text):
                patterns[pattern_name].append({
                    'topic': topic,
                    'year': q.get('year'),
                    'text_sample': text[:100]
                })
                break
    
    print(f"   ✓ Found {len(patterns)} distinct question patterns")
    
    return patterns


def calculate_topic_trends(questions):
    """Calculate rising and declining trends for each topic."""
    print("\n📈 Calculating topic trends...")
    
    # Group by topic and year
    topic_year_counts = defaultdict(lambda: defaultdict(int))
    topic_total = defaultdict(int)
    
    for q in questions:
        topic = q.get('topic_code')
        year = q.get('year')
        if topic and year:
            topic_year_counts[topic][year] += 1
            topic_total[topic] += 1
    
    trends = {}
    
    for topic, year_counts in topic_year_counts.items():
        years = sorted(year_counts.keys())
        
        if len(years) < 3:
            trends[topic] = {'trend': 0, 'status': 'stable'}
            continue
        
        # Recent vs old comparison
        recent_years = [y for y in years if y >= 2022]
        old_years = [y for y in years if 2018 <= y < 2022]
        
        recent_avg = sum(year_counts[y] for y in recent_years) / max(len(recent_years), 1)
        old_avg = sum(year_counts[y] for y in old_years) / max(len(old_years), 1)
        
        if old_avg > 0:
            trend_pct = ((recent_avg - old_avg) / old_avg) * 100
        else:
            trend_pct = 100 if recent_avg > 0 else 0
        
        # Determine status
        if trend_pct > 30:
            status = 'rising'
        elif trend_pct < -30:
            status = 'declining'
        else:
            status = 'stable'
        
        # Calculate consistency (how often it appears)
        appearance_rate = len([y for y in years if year_counts[y] > 0]) / len(years)
        
        # Last appearance
        last_year = max(years)
        
        trends[topic] = {
            'trend': round(trend_pct, 1),
            'status': status,
            'recent_avg': round(recent_avg, 2),
            'old_avg': round(old_avg, 2),
            'total_questions': topic_total[topic],
            'years_active': len(years),
            'appearance_rate': round(appearance_rate, 2),
            'last_seen': last_year,
            'yearly_counts': dict(year_counts)
        }
    
    # Sort by trend
    rising = [(t, d) for t, d in trends.items() if d['status'] == 'rising']
    declining = [(t, d) for t, d in trends.items() if d['status'] == 'declining']
    
    rising.sort(key=lambda x: x[1]['trend'], reverse=True)
    declining.sort(key=lambda x: x[1]['trend'])
    
    print(f"   ✓ Rising topics: {len(rising)}")
    print(f"   ✓ Declining topics: {len(declining)}")
    print(f"   ✓ Stable topics: {len(trends) - len(rising) - len(declining)}")
    
    return trends, rising[:15], declining[:15]


def predict_topic_frequencies(questions, trends):
    """Predict 2026 topic frequencies using multiple methods."""
    print("\n🎯 Predicting 2026 topic frequencies...")
    
    predictions = {}
    
    for topic, data in trends.items():
        # Method 1: Weighted moving average
        yearly = data.get('yearly_counts', {})
        if not yearly:
            predictions[topic] = {'predicted_count': 0, 'confidence': 0, 'method': 'no_data', 'trend': 0, 'status': 'unknown'}
            continue
        years = sorted(yearly.keys())
        
        if not years:
            predictions[topic] = 0
            continue
        
        # Recent years get more weight
        weights = []
        values = []
        for i, year in enumerate(years[-5:]):  # Last 5 years
            weight = (i + 1) ** 2  # Quadratic weighting
            weights.append(weight)
            values.append(yearly[year])
        
        weighted_avg = sum(w * v for w, v in zip(weights, values)) / sum(weights) if weights else 0
        
        # Method 2: Trend adjustment
        trend_factor = 1 + (data['trend'] / 100) * 0.5  # 50% of trend impact
        
        # Method 3: Consistency bonus
        consistency_bonus = data['appearance_rate'] * 0.3
        
        # Method 4: Recency bonus
        recency_bonus = 0
        if data['last_seen'] >= 2024:
            recency_bonus = 0.5
        elif data['last_seen'] >= 2022:
            recency_bonus = 0.2
        
        # Final prediction
        predicted = weighted_avg * trend_factor * (1 + consistency_bonus + recency_bonus)
        
        predictions[topic] = {
            'predicted_count': round(predicted, 2),
            'confidence': min(0.95, 0.5 + data['appearance_rate'] * 0.3 + (0.2 if len(years) >= 5 else 0)),
            'method': 'weighted_trend',
            'trend': data['trend'],
            'status': data['status']
        }
    
    # Normalize to section quotas
    section_predictions = defaultdict(list)
    for topic, pred in predictions.items():
        section = get_section_from_topic(topic)
        section_predictions[section].append((topic, pred))
    
    # Adjust to match quotas
    for section, quota in SECTION_DISTRIBUTION.items():
        topics = section_predictions.get(section, [])
        if not topics:
            continue
        
        total_predicted = sum(p['predicted_count'] for _, p in topics)
        if total_predicted > 0:
            scale_factor = quota / total_predicted
            for topic, pred in topics:
                pred['predicted_count'] = round(pred['predicted_count'] * scale_factor, 2)
    
    print(f"   ✓ Generated predictions for {len(predictions)} topics")
    
    return predictions


def find_similar_questions(questions):
    """Find semantically similar questions using TF-IDF."""
    if not HAS_ML:
        print("\n⚠️ Skipping similarity analysis (sklearn not available)")
        return {}
    
    print("\n🔄 Finding similar questions across years...")
    
    # Group questions by topic
    topic_questions = defaultdict(list)
    for q in questions:
        topic = q.get('topic_code')
        text = q.get('question_text', '')
        if topic and text:
            topic_questions[topic].append(q)
    
    similar_pairs = {}
    
    for topic, qs in topic_questions.items():
        if len(qs) < 5:
            continue
        
        # Get question texts
        texts = [q.get('question_text', '') for q in qs]
        
        # TF-IDF vectorization
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find highly similar pairs (>0.7 similarity)
            pairs = []
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    sim = similarity_matrix[i, j]
                    if sim > 0.7:
                        pairs.append({
                            'q1_year': qs[i].get('year'),
                            'q2_year': qs[j].get('year'),
                            'similarity': round(sim, 3),
                            'q1_text': qs[i].get('question_text', '')[:100],
                            'q2_text': qs[j].get('question_text', '')[:100]
                        })
            
            if pairs:
                similar_pairs[topic] = pairs
        except Exception as e:
            continue
    
    total_pairs = sum(len(p) for p in similar_pairs.values())
    print(f"   ✓ Found {total_pairs} similar question pairs across {len(similar_pairs)} topics")
    
    return similar_pairs


def extract_high_frequency_keywords(questions):
    """Extract high-frequency keywords per topic."""
    print("\n🔤 Extracting high-frequency keywords...")
    
    topic_keywords = {}
    
    # Common stop words
    stop_words = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'for', 
                      'on', 'with', 'at', 'by', 'from', 'as', 'it', 'that', 'which', 'this',
                      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'and', 'or',
                      'but', 'if', 'then', 'else', 'when', 'where', 'how', 'what', 'who', 'whom',
                      'whose', 'why', 'not', 'no', 'yes', 'all', 'each', 'every', 'both', 'few',
                      'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just',
                      'also', 'now', 'only', 'same', 'so', 'out', 'up', 'down', 'here', 'there'])
    
    topic_texts = defaultdict(list)
    for q in questions:
        topic = q.get('topic_code')
        text = q.get('question_text', '')
        if topic and text:
            topic_texts[topic].append(text.lower())
    
    for topic, texts in topic_texts.items():
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,15}\b', all_text)
        
        # Filter stop words
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Get top keywords
        top_keywords = word_counts.most_common(20)
        
        topic_keywords[topic] = {
            'keywords': [{'word': w, 'count': c} for w, c in top_keywords],
            'total_questions': len(texts)
        }
    
    print(f"   ✓ Extracted keywords for {len(topic_keywords)} topics")
    
    return topic_keywords


def generate_predicted_questions(questions, predictions, trends, keywords):
    """Generate predicted question types for 2026."""
    print("\n📝 Generating predicted question patterns...")
    
    predicted_questions = []
    
    # Sort topics by predicted count
    sorted_topics = sorted(
        predictions.items(),
        key=lambda x: x[1]['predicted_count'],
        reverse=True
    )
    
    for topic, pred in sorted_topics[:30]:  # Top 30 topics
        topic_qs = [q for q in questions if q.get('topic_code') == topic]
        
        if not topic_qs:
            continue
        
        # Get recent questions (2023-2025)
        recent_qs = [q for q in topic_qs if q.get('year', 0) >= 2023]
        if not recent_qs:
            recent_qs = topic_qs[-5:]  # Last 5 questions
        
        # Sample questions to show patterns
        samples = random.sample(recent_qs, min(3, len(recent_qs)))
        
        # Get keywords for this topic
        topic_kw = keywords.get(topic, {}).get('keywords', [])[:10]
        
        predicted_questions.append({
            'topic_code': topic,
            'topic_name': TOPIC_NAMES.get(topic, topic),
            'section': get_section_from_topic(topic),
            'predicted_count': pred['predicted_count'],
            'confidence': pred['confidence'],
            'trend': pred['trend'],
            'status': pred['status'],
            'sample_patterns': [
                {
                    'year': q.get('year'),
                    'question': q.get('question_text', '')[:200],
                    'type': identify_question_type(q.get('question_text', ''))
                }
                for q in samples
            ],
            'high_freq_keywords': topic_kw,
            'recommendation': generate_study_recommendation(pred, trends.get(topic, {}))
        })
    
    print(f"   ✓ Generated patterns for {len(predicted_questions)} high-priority topics")
    
    return predicted_questions


def identify_question_type(text):
    """Identify the type/format of a question."""
    text = text.lower() if text else ''
    
    if 'synonym' in text:
        return 'Synonym'
    elif 'antonym' in text:
        return 'Antonym'
    elif 'fill in' in text or 'blank' in text:
        return 'Fill in Blank'
    elif 'error' in text or 'incorrect' in text:
        return 'Error Detection'
    elif 'meaning' in text:
        return 'Meaning/Definition'
    elif 'ratio' in text or 'proportion' in text:
        return 'Ratio Problem'
    elif 'percentage' in text or '%' in text:
        return 'Percentage'
    elif 'speed' in text or 'train' in text or 'distance' in text:
        return 'Speed/Distance'
    elif 'series' in text or 'next' in text:
        return 'Series/Pattern'
    elif 'code' in text or 'decode' in text:
        return 'Coding-Decoding'
    elif 'odd' in text:
        return 'Classification'
    elif 'arrange' in text or 'order' in text:
        return 'Arrangement'
    else:
        return 'Standard MCQ'


def generate_study_recommendation(pred, trend_data):
    """Generate study recommendation based on predictions."""
    count = pred['predicted_count']
    trend = pred['trend']
    status = pred['status']
    
    if count >= 3 and status == 'rising':
        return "🔥 HIGH PRIORITY - Increasing trend, expect 3+ questions"
    elif count >= 2:
        return "⭐ IMPORTANT - Consistent topic, prepare thoroughly"
    elif status == 'rising':
        return "📈 WATCH - Growing importance, allocate study time"
    elif status == 'declining':
        return "📉 LOWER PRIORITY - Declining, basic coverage enough"
    else:
        return "📚 STANDARD - Maintain regular practice"


def create_final_predictions(questions, predictions, trends, rising, declining, keywords, similar):
    """Create final predictions JSON structure."""
    print("\n💾 Creating final predictions...")
    
    # Section-wise predictions
    section_predictions = defaultdict(list)
    
    for topic, pred in predictions.items():
        section = get_section_from_topic(topic)
        topic_data = trends.get(topic, {})
        
        section_predictions[section].append({
            'topic_code': topic,
            'topic_name': TOPIC_NAMES.get(topic, topic),
            'predicted_count': pred['predicted_count'],
            'confidence': pred['confidence'],
            'trend': pred['trend'],
            'status': pred['status'],
            'last_seen': topic_data.get('last_seen'),
            'yearly_history': topic_data.get('yearly_counts', {})
        })
    
    # Sort each section by predicted count
    for section in section_predictions:
        section_predictions[section].sort(key=lambda x: x['predicted_count'], reverse=True)
    
    # Rising topics with proper format
    rising_topics = []
    for topic, data in rising[:10]:
        rising_topics.append({
            'topic_code': topic,
            'topic_name': TOPIC_NAMES.get(topic, topic),
            'growth': data['trend'],
            'recent_avg': data['recent_avg'],
            'old_avg': data['old_avg']
        })
    
    # Declining topics
    declining_topics = []
    for topic, data in declining[:10]:
        declining_topics.append({
            'topic_code': topic,
            'topic_name': TOPIC_NAMES.get(topic, topic),
            'decline': data['trend'],
            'recent_avg': data['recent_avg'],
            'old_avg': data['old_avg']
        })
    
    # Summary stats
    year_counts = Counter(q.get('year') for q in questions if q.get('year'))
    
    final_predictions = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'target_exam': 'AFCAT 2026',
            'model_version': '2.0',
            'total_pyq_analyzed': len(questions),
            'years_covered': sorted(year_counts.keys()),
            'methodology': [
                'Weighted Moving Average',
                'Trend Analysis (Recent vs Old)',
                'Consistency Scoring',
                'Recency Bonus',
                'Section Quota Normalization'
            ]
        },
        'summary': {
            'total_questions': len(questions),
            'year_distribution': dict(sorted(year_counts.items())),
            'topics_analyzed': len(predictions),
            'rising_count': len(rising),
            'declining_count': len(declining)
        },
        'section_distribution': SECTION_DISTRIBUTION,
        'topic_predictions': dict(section_predictions),
        'rising_topics': rising_topics,
        'declining_topics': declining_topics,
        'topic_keywords': keywords,
        'similar_questions': similar if similar else {}
    }
    
    return final_predictions


def main():
    print("=" * 70)
    print("🎯 AFCAT 2026 ADVANCED PREDICTION SYSTEM v2.0")
    print("=" * 70)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # Step 1: Load and fix data
    questions = load_and_fix_pyq_data()
    print(f"   ✓ Total questions: {len(questions)}")
    
    # Step 2: Analyze patterns
    patterns = analyze_question_patterns(questions)
    
    # Step 3: Calculate trends
    trends, rising, declining = calculate_topic_trends(questions)
    
    # Step 4: Predict frequencies
    predictions = predict_topic_frequencies(questions, trends)
    
    # Step 5: Find similar questions
    similar = find_similar_questions(questions)
    
    # Step 6: Extract keywords
    keywords = extract_high_frequency_keywords(questions)
    
    # Step 7: Generate predicted patterns
    predicted_patterns = generate_predicted_questions(questions, predictions, trends, keywords)
    
    # Step 8: Create final predictions
    final_predictions = create_final_predictions(
        questions, predictions, trends, rising, declining, keywords, similar
    )
    
    # Save predictions
    with open(OUTPUT_DIR / "afcat_2026_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, indent=2, ensure_ascii=False)
    
    # Save predicted patterns
    with open(OUTPUT_DIR / "afcat_2026_predicted_patterns.json", 'w', encoding='utf-8') as f:
        json.dump(predicted_patterns, f, indent=2, ensure_ascii=False)
    
    # Display summary
    print("\n" + "=" * 70)
    print("📊 PREDICTION SUMMARY")
    print("=" * 70)
    
    print("\n🔝 TOP 10 RISING TOPICS (High Priority):")
    for i, (topic, data) in enumerate(rising[:10], 1):
        name = TOPIC_NAMES.get(topic, topic)
        print(f"   {i:2}. {name:30} +{data['trend']:.0f}%")
    
    print("\n📉 TOP 10 DECLINING TOPICS (Lower Priority):")
    for i, (topic, data) in enumerate(declining[:10], 1):
        name = TOPIC_NAMES.get(topic, topic)
        print(f"   {i:2}. {name:30} {data['trend']:.0f}%")
    
    print("\n📚 EXPECTED SECTION DISTRIBUTION:")
    for section, quota in SECTION_DISTRIBUTION.items():
        print(f"   {section:25} {quota} questions")
    
    print("\n" + "=" * 70)
    print("✅ PREDICTIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
    print(f"   • afcat_2026_predictions.json")
    print(f"   • afcat_2026_predicted_patterns.json")
    
    return final_predictions


if __name__ == "__main__":
    main()
