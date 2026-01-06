"""
Full API-based Question Classification
=======================================
Extracts ALL questions from PDF and sends each to Groq API for classification.
Returns topic and subject for every question.
"""

import json
import time
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# Groq API settings
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def classify_question_with_api(question_text: str, question_number: int) -> dict:
    """Send a single question to Groq API for classification."""
    
    prompt = f"""You are an expert AFCAT exam classifier. Classify this question.

QUESTION {question_number}:
{question_text}

AFCAT SECTIONS AND TOPICS:

1. VERBAL ABILITY (English):
   - VA_SYN: Synonyms
   - VA_ANT: Antonyms
   - VA_IDIOM: Idioms & Phrases
   - VA_SPELL: Spelling
   - VA_COMP: Reading Comprehension
   - VA_ERR: Error Detection
   - VA_SEN_COMP: Sentence Completion
   - VA_SEN_IMP: Sentence Improvement
   - VA_OWS: One Word Substitution
   - VA_CLOZE: Cloze Test

2. GENERAL AWARENESS:
   - GA_HIST_ANC: Ancient History
   - GA_HIST_MED: Medieval History
   - GA_HIST_MOD: Modern History
   - GA_GEO_IND: Indian Geography
   - GA_GEO_WORLD: World Geography
   - GA_POLITY: Polity & Governance
   - GA_DEF: Defence & Military
   - GA_SPORTS: Sports
   - GA_AWARDS: Awards & Honours
   - GA_ORG: Organizations
   - GA_SCI: Science & Technology
   - GA_CURR: Current Affairs
   - GA_CULTURE: Art & Culture
   - GA_ECON: Economy

3. NUMERICAL ABILITY:
   - NA_PER: Percentage
   - NA_TW: Time & Work
   - NA_SPD: Speed, Distance, Time
   - NA_TRAIN: Trains & Platforms
   - NA_AVG: Average
   - NA_PL: Profit & Loss
   - NA_RAT: Ratio & Proportion
   - NA_NUM: Number System
   - NA_SI: Simple Interest
   - NA_CI: Compound Interest
   - NA_ALG: Algebra
   - NA_GEO: Geometry

4. REASONING & MILITARY APTITUDE:
   - RM_VR_CLASS: Verbal Classification/Odd One Out
   - RM_VR_ANALOGY: Verbal Analogy
   - RM_VR_CODE: Coding-Decoding
   - RM_VR_BLOOD: Blood Relations
   - RM_VR_DIR: Direction Sense
   - RM_VR_SEAT: Seating Arrangement
   - RM_NV_SERIES: Non-Verbal Series
   - RM_NV_PATTERN: Pattern Completion
   - RM_NV_CLASS: Non-Verbal Classification
   - RM_NV_VENN: Venn Diagrams
   - RM_NV_ANALOGY: Non-Verbal Analogy
   - RM_NV_FIG: Figure-based Questions

Respond with ONLY this JSON format (no markdown, no explanation):
{{"section": "verbal_ability|general_awareness|numerical_ability|reasoning", "topic_code": "XX_XXX", "topic_name": "Topic Name", "confidence": 0.XX}}
"""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an AFCAT exam question classifier. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s backoff
                    print(f"   ⏳ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        result = response.json()
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Clean and parse JSON
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "question_number": question_number,
                "section": data.get("section", "unknown"),
                "topic_code": data.get("topic_code", "unknown"),
                "topic_name": data.get("topic_name", "unknown"),
                "confidence": float(data.get("confidence", 0.5)),
                "method": "groq-api"
            }
    except Exception as e:
        print(f"  ⚠️ Q{question_number}: API error - {e}")
    
    return {
        "question_number": question_number,
        "section": "unknown",
        "topic_code": "unknown",
        "topic_name": "unknown",
        "confidence": 0.0,
        "method": "failed"
    }


def extract_and_classify_pdf(pdf_path: str, year: int = None, shift: int = None):
    """Extract questions from PDF and classify each via API."""
    
    print("=" * 70)
    print("FULL API-BASED QUESTION CLASSIFICATION")
    print("=" * 70)
    print(f"\n📄 PDF: {pdf_path}")
    print(f"🤖 Model: {MODEL}")
    print("-" * 70)
    
    # Step 1: Extract questions from PDF
    print("\n📖 Step 1: Extracting questions from PDF...")
    
    from utils.ocr_engine import ExamPaperOCR, MCQExtractor, OCREngine
    
    ocr = ExamPaperOCR(engine=OCREngine.EASYOCR)
    ocr_results = ocr.extract_from_pdf(pdf_path)
    
    full_text = ocr.get_full_text(ocr_results)
    
    mcq_extractor = MCQExtractor()
    questions = mcq_extractor.extract_questions(full_text)
    
    print(f"   ✓ Extracted {len(questions)} questions")
    
    # Step 2: Send each question to Groq API
    print(f"\n🔄 Step 2: Classifying via Groq API ({MODEL})...")
    print("   (This may take a few minutes for 100 questions)\n")
    
    classified = []
    section_counts = {"verbal_ability": 0, "general_awareness": 0, "numerical_ability": 0, "reasoning": 0, "unknown": 0}
    topic_counts = {}
    
    for i, q in enumerate(questions):
        qnum = q.question_number
        qtext = q.text[:500]  # Limit text length
        
        # Rate limiting: ~30 RPM for free tier
        if i > 0:
            time.sleep(2)
        
        result = classify_question_with_api(qtext, qnum)
        result["text"] = qtext
        result["options"] = q.options
        classified.append(result)
        
        # Update counts
        section = result["section"]
        if section in section_counts:
            section_counts[section] += 1
        else:
            section_counts["unknown"] += 1
        
        topic = result["topic_name"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Progress indicator
        status = "✓" if result["method"] == "groq-api" else "⚠️"
        print(f"   {status} Q{qnum:3d}: {section:20s} | {result['topic_code']:15s} | {topic[:30]}")
    
    # Step 3: Generate report
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION RESULTS")
    print("=" * 70)
    
    print(f"\n📁 Section Breakdown:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / len(questions)
            print(f"   {section:25s}: {count:3d} questions ({pct:.1f}%)")
    
    print(f"\n🔝 Top Topics:")
    sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:15]
    for topic, count in sorted_topics:
        print(f"   {topic:35s}: {count:3d}")
    
    # Step 4: Save results
    output_dir = Path("output/api_classification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"afcat_{year or 'unknown'}_{shift or 'unknown'}_api_classified.json"
    output_path = output_dir / filename
    
    output_data = {
        "source": pdf_path,
        "year": year,
        "shift": shift,
        "model": MODEL,
        "total_questions": len(classified),
        "section_breakdown": section_counts,
        "topic_breakdown": topic_counts,
        "questions": classified
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to: {output_path}")
    
    # CSV export
    csv_path = output_dir / filename.replace('.json', '.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Q#,Section,Topic Code,Topic Name,Confidence,Text Preview\n")
        for q in classified:
            text_preview = q['text'][:50].replace(',', ';').replace('\n', ' ')
            f.write(f"{q['question_number']},{q['section']},{q['topic_code']},{q['topic_name']},{q['confidence']:.2f},\"{text_preview}\"\n")
    
    print(f"📄 CSV saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("✅ Classification complete!")
    print("=" * 70)
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full API-based question classification")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--year", type=int, help="Exam year")
    parser.add_argument("--shift", type=int, help="Exam shift")
    
    args = parser.parse_args()
    
    extract_and_classify_pdf(args.pdf, args.year, args.shift)
