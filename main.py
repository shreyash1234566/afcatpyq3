
"""
AFCAT 2026 Prediction System
=============================
Main entry point for the prediction system.

Usage:
    python main.py                    # Generate full prediction
    python main.py --analyze          # Topic analysis only
    python main.py --trends           # Trend analysis
    python main.py --plan             # Generate study plan
    python main.py --news             # Current affairs analysis
    python main.py --export           # Export all reports
    
    # Paper extraction commands
    python main.py --extract paper.pdf           # Extract from PDF
    python main.py --extract paper.jpg           # Extract from image
    python main.py --extract-dir ./papers/       # Batch extract from directory
    python main.py --extract paper.pdf --year 2025 --shift 1  # With metadata
    
    # NEW: Question Bank & AI Generation commands
    python main.py --build-bank              # Build question bank from extracted papers
    python main.py --generate-questions      # Generate predicted questions using Ollama AI
    python main.py --generate-questions --model llama3  # Specify Ollama model
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    EXAM_CONFIG, HISTORICAL_FREQUENCIES, OUTPUT_DIR,
    PREDICTIONS_DIR, REPORTS_DIR, LOGGING_CONFIG
)
from utils.data_structures import Section
from analysis.topic_analyzer import TopicAnalyzer, print_analysis_report
from analysis.trend_detector import TrendDetector, get_hot_topics, get_cold_topics
from models.topic_predictor import TopicPredictor
from models.current_affairs import (
    CurrentAffairsClassifier, create_mock_news_data,
    generate_current_affairs_summary
)
from dashboard import (
    PredictionDashboard, print_prediction_report,
    generate_html_report
)


def setup_logging():
    """Configure logging with UTF-8 support for Windows."""
    import io
    
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create UTF-8 safe stream handler for Windows console
    # This handles emojis and special characters that cp1252 can't encode
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Replace problematic characters on Windows
                stream = self.stream
                try:
                    stream.write(msg + self.terminator)
                except UnicodeEncodeError:
                    # Fallback: encode with 'replace' for unencodable chars
                    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    # Setup handlers
    stream_handler = SafeStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(
        log_dir / f"afcat_prediction_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'  # Explicit UTF-8 encoding for file
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[stream_handler, file_handler]
    )


def create_directories():
    """Create necessary directories."""
    directories = [
        OUTPUT_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        OUTPUT_DIR / "logs",
        BASE_DIR / "data" / "raw",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "data" / "sample"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def run_topic_analysis():
    """Run topic frequency analysis."""
    print("\n" + "=" * 70)
    print("AFCAT 2026 TOPIC FREQUENCY ANALYSIS")
    print("=" * 70)
    
    analyzer = TopicAnalyzer()
    years = list(range(2020, 2026))
    analyzer.load_from_config(HISTORICAL_FREQUENCIES, years)
    
    analysis = analyzer.generate_full_analysis()
    print_analysis_report(analysis)
    
    # Show study priorities
    print("\n" + "=" * 70)
    print("TOP 15 STUDY PRIORITIES (by ROI)")
    print("=" * 70)
    
    priorities = analyzer.get_study_priorities(available_hours=200)
    
    for i, p in enumerate(priorities[:15], 1):
        trend_icon = "📈" if p['trend'] == 'rising' else ("📉" if p['trend'] == 'declining' else "➡️")
        print(f"{i:2}. {p['topic'].replace('_', ' ').title():30} "
              f"| {p['predicted_questions']:4.1f} Qs | "
              f"ROI: {p['roi_score']:5.2f} | "
              f"{p['recommended_hours']:5.1f} hrs {trend_icon}")


def run_trend_analysis():
    """Run trend detection analysis."""
    print("\n" + "=" * 70)
    print("AFCAT 2026 TREND ANALYSIS")
    print("=" * 70)
    
    analyzer = TopicAnalyzer()
    years = list(range(2020, 2026))
    analyzer.load_from_config(HISTORICAL_FREQUENCIES, years)
    
    detector = TrendDetector()
    trend_analysis = detector.analyze_all_trends(analyzer.topic_frequencies)
    
    # Hot topics
    hot_topics = get_hot_topics(trend_analysis, top_n=10)
    print("\n🔥 HOT TOPICS (Rising Trends)")
    print("-" * 50)
    for i, topic in enumerate(hot_topics, 1):
        print(f"{i:2}. {topic['topic'].replace('_', ' ').title():25} "
              f"| Predicted: {topic['predicted']:.1f} Qs | "
              f"Trend: +{topic['trend_strength']:.2f}")
    
    # Cold topics
    cold_topics = get_cold_topics(trend_analysis, top_n=5)
    print("\n❄️ COLD TOPICS (Declining Trends)")
    print("-" * 50)
    for i, topic in enumerate(cold_topics, 1):
        print(f"{i:2}. {topic['topic'].replace('_', ' ').title():25} "
              f"| Predicted: {topic['predicted']:.1f} Qs | "
              f"Decline: -{topic['decline_rate']:.2f}")
    
    # Structural breaks
    print("\n⚠️ STRUCTURAL CHANGES DETECTED")
    print("-" * 50)
    breaks = [
        (topic, data) for topic, data in trend_analysis.items()
        if data['structural_break']['detected']
    ]
    if breaks:
        for topic, data in breaks:
            print(f"   • {topic}: Break in {data['structural_break']['year']} "
                  f"(magnitude: {data['structural_break']['magnitude']:.2f})")
    else:
        print("   No significant structural breaks detected.")
        print("   Note: Math section increased from 18→20 questions in 2024-25.")


def run_current_affairs_analysis():
    """Analyze current affairs relevance."""
    print("\n" + "=" * 70)
    print("AFCAT 2026 CURRENT AFFAIRS ANALYSIS")
    print("=" * 70)
    
    # Use mock data for demo
    articles = create_mock_news_data()
    
    classifier = CurrentAffairsClassifier(use_transformer=False)
    
    print("\n📰 Sample News Classification")
    print("-" * 60)
    
    for article in articles:
        classified = classifier.classify(article)
        print(f"\n📌 {classified.title}")
        print(f"   Category: {classified.category}")
        print(f"   AFCAT Probability: {classified.afcat_probability:.0%}")
        print(f"   Key Facts: {', '.join(classified.key_facts[:3]) if classified.key_facts else 'N/A'}")
    
    print("\n" + "-" * 60)
    print("💡 CURRENT AFFAIRS FOCUS AREAS:")
    print("   1. Defense Operations & Exercises (IAF focus)")
    print("   2. Sports Championships & Trophies")
    print("   3. National Awards (Padma, Gallantry)")
    print("   4. ISRO Missions & Space Technology")
    print("   5. International Summits & Treaties")


def run_full_prediction():
    """Generate full prediction report."""
    print("\n" + "=" * 70)
    print("GENERATING AFCAT 2026 PREDICTIONS...")
    print("=" * 70)
    
    dashboard = PredictionDashboard(target_year=2026)
    prediction = dashboard.generate_full_prediction()
    
    # Print report
    print_prediction_report(prediction)
    
    # Show model metrics
    print("\n📊 MODEL PERFORMANCE")
    print("-" * 40)
    print(f"   Model Type: XGBoost / Gradient Boosting")
    print(f"   Training Data: 2020-2025 (6 years)")
    print(f"   Topics Analyzed: {len(dashboard.analyzer.topic_frequencies)}")
    print(f"   Overall Confidence: {prediction.overall_confidence:.1%}")
    
    return prediction


def export_all_reports():
    """Export all prediction reports."""
    print("\n" + "=" * 70)
    print("EXPORTING ALL REPORTS...")
    print("=" * 70)
    
    dashboard = PredictionDashboard(target_year=2026)
    
    # Export predictions
    files = dashboard.export_predictions()
    
    print(f"\n✅ Prediction JSON: {files['prediction']}")
    print(f"✅ Blueprint JSON: {files['blueprint']}")
    print(f"✅ Study Plan JSON: {files['study_plan']}")
    
    # Generate HTML report
    prediction = dashboard.generate_full_prediction()
    html_path = REPORTS_DIR / "afcat_2026_report.html"
    generate_html_report(prediction, html_path)
    print(f"✅ HTML Report: {html_path}")
    
    print("\n" + "=" * 70)
    print("All reports exported successfully!")
    print("=" * 70)


def run_paper_extraction(file_path: str, year: int = None, shift: int = None, single_section: str = None, zone_mode: str = None):
    """Extract and classify questions from exam paper."""
    print("\n" + "=" * 70)
    print("EXTRACTING QUESTIONS FROM EXAM PAPER")
    print("=" * 70)
    
    try:
        from pipeline import AFCATExamAnalyzer
        from utils.ocr_engine import OCREngine
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nTo use paper extraction, install:")
        print("  pip install easyocr pymupdf opencv-python")
        print("\nFor better accuracy (optional):")
        print("  pip install paddlepaddle paddleocr")
        return None
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"\n❌ File not found: {file_path}")
        return None
    
    print(f"\n📄 Source: {file_path.name}")
    print(f"📅 Year: {year or 'Auto-detect'}")
    print(f"🔢 Shift: {shift or 'Unknown'}")
    print("-" * 50)
    
    # Initialize analyzer
    try:
        analyzer = AFCATExamAnalyzer(
            ocr_engine=OCREngine.EASYOCR,
            use_transformers=False,  # Keyword-based for speed
            gpu=False
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize OCR: {e}")
        print("\nTry installing: pip install easyocr")
        return None
    
    # Run analysis
    try:
        result = analyzer.analyze_paper(
            file_path,
            year=year,
            shift=shift,
            zone_mode=zone_mode,
            single_section=single_section
        )
        
        # Show review report
        review = analyzer.get_review_report(result)
        if review['needs_review'] > 0:
            print(f"\n⚠️ {review['needs_review']} questions need manual review")
            print("   Run with --verbose to see details")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_extraction(directory: str):
    """Extract questions from all papers in a directory."""
    print("\n" + "=" * 70)
    print("BATCH EXTRACTING FROM DIRECTORY")
    print("=" * 70)
    
    try:
        from pipeline import AFCATExamAnalyzer
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("Install with: pip install easyocr pymupdf opencv-python")
        return None
    
    directory = Path(directory)
    
    if not directory.exists():
        print(f"\n❌ Directory not found: {directory}")
        return None
        
    if not directory.is_dir():
        print(f"\n❌ Not a directory: {directory}")
        return None
    
    # List files
    pdf_files = list(directory.glob("*.pdf"))
    image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))
    all_files = pdf_files + image_files
    
    if not all_files:
        print(f"\n❌ No PDF or image files found in: {directory}")
        return None
        
    print(f"\n📁 Directory: {directory}")
    print(f"📄 Found {len(all_files)} files ({len(pdf_files)} PDFs, {len(image_files)} images)")
    print("-" * 50)
    
    # Process each file
    analyzer = AFCATExamAnalyzer(use_transformers=False)
    results = analyzer.analyze_batch(directory)
    
    print(f"\n✅ Processed {len(results)} papers")
    
    # Summary
    total_questions = sum(r.total_questions for r in results)
    print(f"📊 Total questions extracted: {total_questions}")
    
    return results


def run_build_question_bank():
    """Build question bank from all extracted analysis files."""
    print("\n" + "=" * 70)
    print("BUILDING QUESTION BANK DATABASE")
    print("=" * 70)
    
    try:
        from data.question_bank import QuestionBankDB, import_analysis_files
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return None
    
    # Initialize database
    db = QuestionBankDB()
    
    # Import from processed analysis files
    analysis_dir = BASE_DIR / "data" / "processed"
    
    print(f"\n📁 Scanning: {analysis_dir}")
    
    if not analysis_dir.exists():
        print(f"\n⚠️ No processed data found at {analysis_dir}")
        print("   Run --extract or --extract-dir first to process exam papers.")
        
        # Still show current stats
        stats = db.get_statistics()
        if stats['total_questions'] > 0:
            print(f"\n📊 Current Question Bank:")
            print(f"   Total Questions: {stats['total_questions']}")
            print(f"   Unique Topics: {stats['unique_topics']}")
        return db
    
    # Import all analysis files
    stats = import_analysis_files(db, analysis_dir)
    
    print(f"\n✅ Import Complete:")
    print(f"   Files Processed: {stats['files']}")
    print(f"   Questions Added: {stats['added']}")
    print(f"   Duplicates Skipped: {stats['duplicates']}")
    
    # Show current stats
    db_stats = db.get_statistics()
    print(f"\n📊 Question Bank Stats:")
    print(f"   Total Questions: {db_stats['total_questions']}")
    print(f"   Unique Topics: {db_stats['unique_topics']}")
    print(f"   Years Covered: {db_stats['years_covered']}")
    
    # Export topic JSON files for HTML dropdown
    topic_json_dir = REPORTS_DIR / "topic_questions"
    count = db.export_topic_questions_for_html(topic_json_dir)
    print(f"\n📁 Exported {count} topic JSON files to: {topic_json_dir}")
    
    return db


def run_generate_questions(model: str = "llama3", topics_per_section: int = 3, questions_per_topic: int = 2):
    """Generate predicted questions using Ollama AI."""
    print("\n" + "=" * 70)
    print("AI QUESTION GENERATION (Ollama)")
    print("=" * 70)
    
    try:
        from data.question_bank import QuestionBankDB
        from models.question_generator import AFCATQuestionGenerator, export_generated_questions
        from analysis.question_patterns import QuestionPatternAnalyzer
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return None
    
    # Initialize
    db = QuestionBankDB()
    generator = AFCATQuestionGenerator(db, ollama_model=model)
    
    # Check Ollama availability
    print(f"\n🔍 Checking Ollama ({model})...")
    
    if generator.ollama.is_available():
        models = generator.ollama.list_models()
        print(f"✅ Ollama connected! Available models: {models}")
    else:
        print("⚠️ Ollama not available. Starting Ollama server is recommended:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Run: ollama serve")
        print("   3. Pull model: ollama pull llama3")
        print("\n   Will use template-based fallback generation...")
    
    # Get prediction data to know which topics to focus on
    dashboard = PredictionDashboard(target_year=2026)
    prediction = dashboard.generate_full_prediction()
    
    print(f"\n📝 Generating questions for top predicted topics...")
    print("-" * 50)
    
    # Prepare predictions by section
    predictions_by_section = {}
    for section, sec_pred in prediction.section_predictions.items():
        predictions_by_section[section.value] = [
            {
                'topic': tp.topic,
                'confidence': tp.confidence,
                'predicted_count': tp.predicted_count
            }
            for tp in sec_pred.topic_predictions[:topics_per_section]
        ]
    
    # Generate questions
    results = generator.generate_for_predictions(
        predictions_by_section,
        questions_per_topic=questions_per_topic
    )
    
    # Summary
    total_generated = 0
    print("\n✅ Generation Complete:")
    for section, gen_results in results.items():
        section_qs = sum(len(r.questions) for r in gen_results)
        total_generated += section_qs
        print(f"   {section.replace('_', ' ').title()}: {section_qs} questions")
    
    # Export results
    output_json = PREDICTIONS_DIR / "generated_questions.json"
    output_html = REPORTS_DIR / "predicted_questions.html"
    
    export_generated_questions(results, output_json, format="json")
    export_generated_questions(results, output_html, format="html")
    
    print(f"\n📁 Saved to:")
    print(f"   JSON: {output_json}")
    print(f"   HTML: {output_html}")
    
    # Show sample
    print("\n" + "-" * 50)
    print("📋 SAMPLE GENERATED QUESTIONS:")
    print("-" * 50)
    
    shown = 0
    for section, gen_results in results.items():
        for result in gen_results:
            for q in result.questions[:1]:  # Show 1 per topic
                if shown < 3:  # Show max 3 samples
                    print(f"\n🎯 Topic: {result.topic.replace('_', ' ').title()} ({result.target_difficulty})")
                    print(f"   Q: {q.question_text[:100]}...")
                    shown += 1
    
    return results


def export_all_reports_with_bank():
    """Export all prediction reports with question bank integration."""
    print("\n" + "=" * 70)
    print("EXPORTING ALL REPORTS (WITH QUESTION BANK)")
    print("=" * 70)
    
    dashboard = PredictionDashboard(target_year=2026)
    
    # Export predictions
    files = dashboard.export_predictions()
    
    print(f"\n✅ Prediction JSON: {files['prediction']}")
    print(f"✅ Blueprint JSON: {files['blueprint']}")
    print(f"✅ Study Plan JSON: {files['study_plan']}")
    
    # Load question bank data for HTML
    question_bank_data = {}
    try:
        from data.question_bank import QuestionBankDB
        db = QuestionBankDB()
        topics = db.get_all_topics()
        
        for topic_info in topics:
            topic = topic_info['topic']
            questions = db.get_questions_by_topic(topic, limit=10)
            question_bank_data[topic] = [
                {
                    'text': q.question_text,
                    'options': q.options,
                    'year': q.year,
                    'shift': q.shift,
                    'difficulty': q.difficulty,
                    'question_type': q.question_type
                }
                for q in questions
            ]
        print(f"✅ Question Bank loaded: {sum(len(qs) for qs in question_bank_data.values())} questions")
    except Exception as e:
        print(f"⚠️ Question Bank not available: {e}")
        print("   Run --build-bank first to enable question dropdowns.")
    
    # Generate HTML report with question bank
    prediction = dashboard.generate_full_prediction()
    html_path = REPORTS_DIR / "afcat_2026_report.html"
    generate_html_report(prediction, html_path, question_bank_data=question_bank_data)
    print(f"✅ HTML Report: {html_path}")
    
    print("\n" + "=" * 70)
    print("All reports exported successfully!")
    print("=" * 70)


def print_disclaimer():
    """Print important disclaimer."""
    print("\n" + "⚠️ " * 20)
    print("\n⚠️  IMPORTANT DISCLAIMER  ⚠️")
    print("-" * 60)
    print("""
This system predicts TOPIC PATTERNS and DISTRIBUTIONS, NOT exact questions.

Expected Accuracy:
  • Topic Distribution: 65-75% (which topics appear)
  • Question Count per Topic: ±1-2 questions
  • Difficulty Level: 55-60%
  • Specific Questions: 25-40% (similar patterns only)

This is a PREPARATION OPTIMIZATION tool, not a crystal ball.
Use predictions to GUIDE study priorities, not as guarantees.

The exam board deliberately introduces variability to prevent
predictability. Always prepare broadly within the syllabus.
""")
    print("⚠️ " * 20 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AFCAT 2026 Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                Generate full prediction
  python main.py --analyze      Topic analysis only
  python main.py --trends       Trend analysis
  python main.py --news         Current affairs analysis
  python main.py --export       Export all reports

Paper Extraction:
  python main.py --extract afcat_2025.pdf              Extract from PDF
  python main.py --extract afcat_2025.pdf --year 2025  With year metadata
  python main.py --extract-dir ./papers/               Batch extract all

Question Bank & AI Generation:
  python main.py --build-bank                          Build question database
  python main.py --generate-questions                  Generate predicted questions
  python main.py --generate-questions --model mistral  Use specific Ollama model
        """
    )
    
    parser.add_argument('--analyze', action='store_true',
                       help='Run topic frequency analysis')
    parser.add_argument('--trends', action='store_true',
                       help='Run trend detection analysis')
    parser.add_argument('--news', action='store_true',
                       help='Run current affairs analysis')
    parser.add_argument('--export', action='store_true',
                       help='Export all reports to files')
    
    # Paper extraction arguments
    parser.add_argument('--extract', type=str, metavar='FILE',
                       help='Extract questions from PDF or image file')
    parser.add_argument('--extract-dir', type=str, metavar='DIR',
                       help='Batch extract from directory')
    parser.add_argument('--year', type=int,
                       help='Exam year (for extraction)')
    parser.add_argument('--shift', type=int,
                       help='Exam shift number (for extraction)')
    parser.add_argument('--zone-mode', type=str,
                       choices=['strict', 'flex', 'off'],
                       default='flex',
                       help='Zone enforcement mode for >=80Q papers (strict|flex|off). Default: flex')
    parser.add_argument('--single-section', type=str,
                       choices=['verbal_ability', 'general_awareness', 'reasoning', 'numerical_ability'],
                       help='Force all questions into one section for short (<80Q) papers')
    
    # Question Bank & Generation arguments
    parser.add_argument('--build-bank', action='store_true',
                       help='Build question bank from extracted papers')
    parser.add_argument('--generate-questions', action='store_true',
                       help='Generate predicted questions using Ollama AI')
    parser.add_argument('--model', type=str, default='llama3',
                       help='Ollama model to use (default: llama3)')
    parser.add_argument('--topics-per-section', type=int, default=3,
                       help='Number of topics per section to generate for (default: 3)')
    parser.add_argument('--questions-per-topic', type=int, default=2,
                       help='Number of questions per topic (default: 2)')
    
    parser.add_argument('--no-disclaimer', action='store_true',
                       help='Skip disclaimer message')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Setup
    if not args.quiet:
        setup_logging()
    create_directories()
    
    # Print header
    print("\n" + "🎯 " * 20)
    print("\n" + " " * 20 + "AFCAT 2026 PREDICTION SYSTEM")
    print(" " * 20 + "Data Science Approach to Exam Preparation")
    print("\n" + "🎯 " * 20)
    
    # Print disclaimer
    if not args.no_disclaimer and not args.extract and not args.extract_dir and not args.build_bank and not args.generate_questions:
        print_disclaimer()
    
    # Run requested analysis
    if args.extract:
        run_paper_extraction(
            args.extract,
            year=args.year,
            shift=args.shift,
            single_section=args.single_section,
            zone_mode=args.zone_mode,
        )
    elif args.extract_dir:
        run_batch_extraction(args.extract_dir)
    elif args.build_bank:
        run_build_question_bank()
    elif args.generate_questions:
        run_generate_questions(
            model=args.model,
            topics_per_section=args.topics_per_section,
            questions_per_topic=args.questions_per_topic
        )
    elif args.analyze:
        run_topic_analysis()
    elif args.trends:
        run_trend_analysis()
    elif args.news:
        run_current_affairs_analysis()
    elif args.export:
        export_all_reports_with_bank()
    else:
        # Default: full prediction
        prediction = run_full_prediction()
        
        # Quick export
        print("\n💾 To export reports, run: python main.py --export")
        print("📄 To extract from papers, run: python main.py --extract <file>")
        print("📚 To build question bank, run: python main.py --build-bank")
        print("🤖 To generate predicted questions, run: python main.py --generate-questions")
    
    print("\n✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
