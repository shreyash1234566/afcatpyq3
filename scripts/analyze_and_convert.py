#!/usr/bin/env python3
"""
Complete Q.json Analysis & Conversion Pipeline
Analyzes existing question data and converts to training format
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ANALYZER
# ============================================================================

class QJsonAnalyzer:
    """Analyze JSON question files for structure and compatibility."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
    
    def analyze(self, json_path: str) -> Dict:
        """Analyze JSON file structure and content."""
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine format
        if isinstance(data, list):
            items = data
            format_type = "Array"
        elif isinstance(data, dict) and 'questions' in data:
            items = data['questions']
            format_type = "Object with 'questions' key"
        else:
            items = [data]
            format_type = "Single Object"
        
        # Analyze structure
        all_keys = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        # Categorize fields
        analysis = {
            'format': format_type,
            'total_items': len(items),
            'all_keys': sorted(all_keys),
            'sample_item': items[0] if items else {},
            'field_frequency': {},
            'sections': Counter(),
            'topics': Counter(),
            'difficulties': Counter(),
            'data_quality': {
                'empty_text': 0,
                'missing_section': 0,
                'missing_topic': 0,
                'missing_options': 0
            }
        }
        
        # Analyze each item
        for item in items:
            if not isinstance(item, dict):
                continue
            
            # Field frequency
            for key in all_keys:
                if key in item:
                    analysis['field_frequency'][key] = analysis['field_frequency'].get(key, 0) + 1
            
            # Content analysis
            text = item.get('text', item.get('question_text', '')).strip()
            if not text:
                analysis['data_quality']['empty_text'] += 1
            
            section = item.get('section', '')
            if not section:
                analysis['data_quality']['missing_section'] += 1
            else:
                analysis['sections'][section] += 1
            
            topic = item.get('topic', item.get('topic_code', ''))
            if not topic:
                analysis['data_quality']['missing_topic'] += 1
            else:
                analysis['topics'][topic] += 1
            
            difficulty = item.get('difficulty', '')
            if difficulty:
                analysis['difficulties'][difficulty] += 1
            
            options = item.get('options', item.get('choices', []))
            if not options:
                analysis['data_quality']['missing_options'] += 1
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print analysis results."""
        
        print("\n" + "=" * 80)
        print("📊 JSON STRUCTURE ANALYSIS")
        print("=" * 80)
        
        print(f"\n📋 Format: {analysis['format']}")
        print(f"📈 Total Items: {analysis['total_items']}")
        
        print(f"\n🔑 Fields Found: {', '.join(analysis['all_keys'][:10])}")
        if len(analysis['all_keys']) > 10:
            print(f"   ... and {len(analysis['all_keys']) - 10} more")
        
        print(f"\n📁 Field Frequency:")
        for field, freq in sorted(analysis['field_frequency'].items(), key=lambda x: -x[1]):
            pct = (freq / analysis['total_items'] * 100)
            print(f"   {field:<25}: {freq:>4} ({pct:>5.1f}%)")
        
        print(f"\n📚 Section Distribution:")
        for section, count in sorted(analysis['sections'].items(), key=lambda x: -x[1]):
            pct = (count / analysis['total_items'] * 100)
            print(f"   {section:<30}: {count:>4} ({pct:>5.1f}%)")
        
        print(f"\n🎯 Top 10 Topics:")
        for topic, count in analysis['topics'].most_common(10):
            print(f"   {topic:<30}: {count:>4}")
        
        if analysis['difficulties']:
            print(f"\n📈 Difficulty Distribution:")
            for diff, count in sorted(analysis['difficulties'].items()):
                pct = (count / analysis['total_items'] * 100)
                print(f"   {diff:<30}: {count:>4} ({pct:>5.1f}%)")
        
        print(f"\n🔎 Data Quality:")
        for metric, count in analysis['data_quality'].items():
            pct = (count / analysis['total_items'] * 100) if analysis['total_items'] > 0 else 0
            status = "❌" if count > 0 else "✅"
            print(f"   {status} {metric:<25}: {count:>4} ({pct:>5.1f}%)")
        
        # Compatibility check
        print(f"\n✅ Compatibility Check:")
        required = ['text', 'section', 'topic']
        has_all = all(any(f in analysis['all_keys'] for f in [k, 'question_text', 'text'] 
                          if k == 'text') for k in required)
        
        if 'text' in analysis['all_keys'] or 'question_text' in analysis['all_keys']:
            print(f"   ✓ Has question text")
        else:
            print(f"   ✗ Missing question text")
        
        if 'section' in analysis['all_keys']:
            print(f"   ✓ Has section field")
        else:
            print(f"   ✗ Missing section field")
        
        if 'topic' in analysis['all_keys'] or 'topic_code' in analysis['all_keys']:
            print(f"   ✓ Has topic field")
        else:
            print(f"   ✗ Missing topic field")
        
        print("=" * 80)


# ============================================================================
# CONVERTER
# ============================================================================

class QJsonConverter:
    """Convert Q.json to standardized training format."""
    
    SECTION_MAPPING = {
        'verbal_ability': 'Verbal Ability',
        'verbal ability': 'Verbal Ability',
        'va': 'Verbal Ability',
        'english': 'Verbal Ability',
        'general_awareness': 'General Awareness',
        'general awareness': 'General Awareness',
        'ga': 'General Awareness',
        'gk': 'General Awareness',
        'numerical_ability': 'Numerical Ability',
        'numerical ability': 'Numerical Ability',
        'na': 'Numerical Ability',
        'math': 'Numerical Ability',
        'maths': 'Numerical Ability',
        'reasoning': 'Reasoning',
        'reasoning & military aptitude': 'Reasoning',
        'rm': 'Reasoning',
    }
    
    TOPIC_CODE_MAPPING = {
        'synonyms': 'VA_SYN',
        'antonyms': 'VA_ANT',
        'comprehension': 'VA_COMP',
        'cloze': 'VA_CLOZE',
        'error': 'VA_ERR',
        'sentence': 'VA_SENT',
        'vocabulary': 'VA_SYN',
        'grammar': 'VA_GRAM',
        
        'history': 'GA_HIST_MOD',
        'geography': 'GA_GEO_IND',
        'polity': 'GA_POLITY',
        'economy': 'GA_ECON',
        'environment': 'GA_ENV',
        'science': 'GA_SCI',
        'defence': 'GA_DEF',
        'current': 'GA_CURR',
        'sports': 'GA_SPORTS',
        'awards': 'GA_AWARDS',
        
        'number': 'NA_NUM',
        'percentage': 'NA_PER',
        'profit': 'NA_PL',
        'interest': 'NA_SI',
        'time': 'NA_TW',
        'speed': 'NA_SPD',
        'ratio': 'NA_RAT',
        'average': 'NA_AVG',
        'algebra': 'NA_ALG',
        
        'analogy': 'RM_VR_ANALOGY',
        'classification': 'RM_VR_CLASS',
        'series': 'RM_VR_SERIES',
        'pattern': 'RM_NV_PATTERN',
        'reasoning': 'RM_VR_ANALOGY',
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
    
    def convert(
        self,
        input_json_path: str,
        year: int,
        shift: int,
        output_json_path: Optional[str] = None,
        output_training_path: Optional[str] = None
    ) -> tuple:
        """Convert Q.json to standardized format."""
        
        # Load input
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle formats
        if isinstance(data, dict) and 'questions' in data:
            items = data['questions']
        elif isinstance(data, list):
            items = data
        else:
            items = [data]
        
        # Convert items
        converted = []
        for idx, item in enumerate(items, 1):
            q = self._convert_item(item, year, shift, idx)
            if q:
                converted.append(q)
        
        # Standard format
        standard = {
            "metadata": {
                "year": year,
                "shift": shift,
                "total_questions": len(converted),
                "source": input_json_path
            },
            "questions": converted
        }
        
        # Save standard
        if not output_json_path:
            output_json_path = f"data/ground_truth/afcat_{year}_{shift}.json"
        
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(standard, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n✅ Saved standard format: {output_json_path}")
        
        # Training format
        training = {
            "metadata": {
                "year": year,
                "shift": shift,
                "total_samples": len(converted)
            },
            "samples": []
        }
        
        for q in converted:
            choices_text = " | ".join(
                [f"{i+1}) {opt}" for i, opt in enumerate(q.get('choices', []))]
            )
            
            training['samples'].append({
                "id": f"{year}_{shift}_Q{q['question_number']}",
                "text": q['question_text'],
                "section": q['section'],
                "topic_code": q['topic_code'],
                "difficulty": q.get('difficulty', 'Medium'),
                "choices_text": choices_text,
                "correct_answer": q.get('correct_answer', ''),
                "explanation": q.get('explanation', '')
            })
        
        # Save training
        if not output_training_path:
            output_training_path = f"data/training/afcat_{year}_{shift}_training.json"
        
        Path(output_training_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_training_path, 'w', encoding='utf-8') as f:
            json.dump(training, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"✅ Saved training format: {output_training_path}")
        
        self._print_summary(standard)
        
        return standard, training
    
    def _convert_item(self, item: dict, year: int, shift: int, idx: int) -> Optional[dict]:
        """Convert single item."""
        
        q_num = item.get('question_number', item.get('id', idx))
        if isinstance(q_num, str):
            try:
                q_num = int(q_num.split('_')[-1])
            except:
                q_num = idx
        
        # Text
        q_text = (item.get('text') or item.get('question_text') or '').strip()
        if not q_text:
            self.warnings.append(f"Q{q_num}: Missing text")
            return None
        
        # Section
        section = (item.get('section') or '').strip()
        if not section:
            self.errors.append(f"Q{q_num}: Missing section")
            return None
        
        section = self._normalize_section(section)
        
        # Topic
        topic = (item.get('topic') or item.get('topic_code') or '').strip()
        if not topic:
            self.warnings.append(f"Q{q_num}: Missing topic")
            topic = "general"
        
        topic_code = self._normalize_topic_code(topic, section)
        
        # Options
        options = item.get('options') or item.get('choices') or []
        if not options:
            self.errors.append(f"Q{q_num}: Missing options")
            return None
        
        # Convert options to list of strings
        if isinstance(options, list):
            if options and isinstance(options[0], dict):
                options = [o.get('text', o.get('value', '')) for o in options]
            options = [str(o).strip() for o in options]
        
        return {
            'question_number': q_num,
            'question_text': q_text,
            'section': section,
            'topic': topic,
            'topic_code': topic_code,
            'choices': options,
            'correct_answer': item.get('correct_answer') or item.get('answer') or '',
            'difficulty': (item.get('difficulty') or 'Medium').capitalize(),
            'explanation': item.get('explanation') or '',
            'year': year,
            'shift': shift
        }
    
    def _normalize_section(self, section: str) -> str:
        """Normalize section name."""
        normalized = self.SECTION_MAPPING.get(section.lower().strip(), section)
        return normalized
    
    def _normalize_topic_code(self, topic: str, section: str) -> str:
        """Generate topic code from topic name."""
        
        topic_lower = topic.lower().strip()
        
        # Try direct mapping
        for key, code in self.TOPIC_CODE_MAPPING.items():
            if key in topic_lower:
                return code
        
        # Fallback based on section
        section_prefix = {
            'Verbal Ability': 'VA',
            'General Awareness': 'GA',
            'Numerical Ability': 'NA',
            'Reasoning': 'RM'
        }.get(section, 'XX')
        
        # Create code from topic
        topic_short = topic_lower.replace(' ', '_')[:10]
        return f"{section_prefix}_{topic_short.upper()}"
    
    def _print_summary(self, data: dict):
        """Print summary."""
        
        questions = data['questions']
        
        print("\n" + "=" * 80)
        print("✅ CONVERSION SUMMARY")
        print("=" * 80)
        
        # Sections
        sections = {}
        for q in questions:
            sec = q.get('section')
            sections[sec] = sections.get(sec, 0) + 1
        
        print(f"\n📁 Section Distribution:")
        for sec, count in sorted(sections.items()):
            pct = (count / len(questions) * 100)
            print(f"   {sec:<30}: {count:>4} ({pct:>5.1f}%)")
        
        # Topics
        topics = {}
        for q in questions:
            topic = q.get('topic_code')
            topics[topic] = topics.get(topic, 0) + 1
        
        print(f"\n🎯 Topic Distribution (Top 10):")
        for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:10]:
            print(f"   {topic:<30}: {count:>4}")
        
        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
        
        if self.warnings:
            print(f"\n⚠️ Warnings: {len(self.warnings)}")
        
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run full analysis and conversion pipeline."""
    
    input_file = "data/sample/sample_questions.json"
    year = 2020
    shift = 1
    
    print("\n" + "=" * 80)
    print("🚀 Q.JSON ANALYSIS & CONVERSION PIPELINE")
    print("=" * 80)
    
    # Step 1: Analyze
    print(f"\n📖 Step 1: Analyzing {input_file}...")
    analyzer = QJsonAnalyzer(verbose=True)
    analysis = analyzer.analyze(input_file)
    analyzer.print_analysis(analysis)
    
    # Step 2: Convert
    print(f"\n📝 Step 2: Converting to training format...")
    converter = QJsonConverter(verbose=True)
    standard, training = converter.convert(
        input_file,
        year,
        shift,
        output_json_path=f"data/ground_truth/afcat_{year}_{shift}.json",
        output_training_path=f"data/training/afcat_{year}_{shift}_training.json"
    )
    
    print(f"\n✅ Pipeline complete!")
    print(f"   - Standard format: data/ground_truth/afcat_{year}_{shift}.json")
    print(f"   - Training format: data/training/afcat_{year}_{shift}_training.json")


if __name__ == "__main__":
    main()
