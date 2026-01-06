"""
AFCAT Prediction System Configuration
=====================================
Centralized configuration for all prediction parameters.
"""

from pathlib import Path
from datetime import datetime

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"

# =============================================================================
# EXAM CONFIGURATION - AFCAT 2026
# =============================================================================
EXAM_CONFIG = {
    "target_year": 2026,
    "total_questions": 100,
    "total_marks": 300,
    "duration_minutes": 120,
    "correct_marks": 3,
    "negative_marks": -1,
    
    # Section-wise distribution (updated for 2026)
    "sections": {
        "verbal_ability": {
            "name": "Verbal Ability in English",
            "questions": 30,
            "marks": 90
        },
        "general_awareness": {
            "name": "General Awareness",
            "questions": 25,
            "marks": 75
        },
        "reasoning": {
            "name": "Reasoning & Military Aptitude",
            "questions": 25,
            "marks": 75
        },
        "numerical_ability": {
            "name": "Numerical Ability",
            "questions": 20,  # Increased from 18 in 2024
            "marks": 60
        }
    }
}

# =============================================================================
# TOPIC TAXONOMY - Based on Historical Analysis
# =============================================================================
TOPIC_TAXONOMY = {
    "verbal_ability": {
        "reading_comprehension": ["passage_based", "inference", "vocabulary_in_context"],
        "cloze_test": ["grammar_based", "vocabulary_based", "mixed"],
        "synonyms": ["olq_vocabulary", "common_words", "contextual"],
        "antonyms": ["olq_vocabulary", "common_words", "contextual"],
        "error_spotting": ["tenses", "prepositions", "subject_verb", "articles"],
        "sentence_completion": ["grammar", "vocabulary", "idiom_based"],
        "idioms_phrases": ["animal_based", "body_part_based", "action_based"],
        "one_word_substitution": ["person", "place", "action", "quality"],
        "para_jumbles": ["sentence_order", "paragraph_order"]
    },
    
    "general_awareness": {
        "defense": ["operations", "missiles", "equipment", "ranks", "exercises"],
        "current_affairs": ["national", "international", "sports_events", "awards"],
        "history": ["modern_india", "freedom_struggle", "medieval", "ancient"],
        "geography": ["physical", "rivers", "mountains", "passes", "climate"],
        "polity": ["constitution", "governance", "schemes"],
        "science": ["physics", "chemistry", "biology", "technology"],
        "sports": ["terms", "trophies", "venues", "records"],
        "economy": ["budget", "schemes", "organizations"],
        "static_gk": ["capitals", "currencies", "firsts", "superlatives"]
    },
    
    "reasoning": {
        "spatial_ability": ["embedded_figures", "rotated_blocks", "mirror_image", "paper_folding"],
        "dot_situation": ["position_analysis", "region_identification"],
        "venn_diagrams": ["two_set", "three_set", "classification"],
        "analogy": ["word_analogy", "number_analogy", "semantic"],
        "series": ["number_series", "letter_series", "mixed_series"],
        "coding_decoding": ["letter_shift", "symbol_based", "mixed"],
        "blood_relations": ["simple", "complex", "coded"],
        "direction_sense": ["simple", "complex"],
        "odd_one_out": ["word_based", "number_based", "logic_based"],
        "syllogism": ["two_statement", "three_statement"]
    },
    
    "numerical_ability": {
        "speed_time_distance": ["trains", "boats_streams", "relative_speed", "races"],
        "time_and_work": ["efficiency", "pipes_cisterns", "men_days"],
        "profit_loss": ["successive_discount", "dishonest_dealer", "marked_price"],
        "simple_compound_interest": ["si_ci_difference", "installments", "compound_periods"],
        "ratio_proportion": ["simple_ratio", "compound_ratio", "partnership"],
        "percentage": ["increase_decrease", "successive", "population"],
        "average": ["weighted", "running", "age_based"],
        "decimal_fractions": ["simplification", "bodmas", "conversion"],
        "number_system": ["lcm_hcf", "divisibility", "remainders"],
        "clocks": ["angle", "overlap", "gain_loss"],
        "probability": ["basic", "conditional", "permutation_combination"],
        "mensuration": ["area", "volume", "surface_area"],
        "algebra": ["equations", "quadratic", "inequalities"]
    }
}

# =============================================================================
# HISTORICAL TOPIC FREQUENCIES (2020-2025 Data)
# =============================================================================
# Format: {topic: [year2020, year2021, year2022, year2023, year2024, year2025]}
HISTORICAL_FREQUENCIES = {
    # Numerical Ability
    "speed_time_distance": [3, 4, 3, 4, 3, 4],
    "time_and_work": [2, 3, 2, 3, 3, 3],
    "profit_loss": [2, 2, 3, 2, 2, 3],
    "simple_compound_interest": [2, 1, 2, 2, 1, 2],
    "ratio_proportion": [2, 2, 1, 2, 2, 2],
    "percentage": [2, 2, 2, 1, 2, 2],
    "average": [1, 2, 2, 2, 1, 1],
    "decimal_fractions": [2, 2, 3, 2, 2, 2],
    "clocks": [0, 1, 0, 1, 1, 1],
    "probability": [0, 0, 0, 0, 1, 1],
    
    # Verbal Ability
    "reading_comprehension": [4, 3, 4, 4, 3, 4],
    "cloze_test": [5, 6, 5, 6, 5, 6],
    "synonyms": [4, 5, 5, 5, 5, 5],
    "antonyms": [3, 4, 4, 3, 4, 4],
    "error_spotting": [3, 3, 4, 3, 3, 4],
    "idioms_phrases": [3, 4, 3, 4, 3, 3],
    "sentence_completion": [3, 2, 3, 2, 3, 2],
    "one_word_substitution": [2, 2, 2, 2, 2, 2],
    "para_jumbles": [2, 1, 2, 2, 2, 2],
    
    # Reasoning
    "spatial_ability": [5, 6, 5, 6, 5, 6],
    "dot_situation": [2, 3, 2, 3, 2, 3],
    "venn_diagrams": [2, 3, 3, 2, 3, 2],
    "analogy": [4, 4, 5, 4, 4, 5],
    "series": [3, 3, 3, 3, 3, 3],
    "coding_decoding": [2, 2, 2, 2, 2, 2],
    "blood_relations": [2, 1, 2, 1, 2, 1],
    "direction_sense": [1, 2, 1, 2, 1, 2],
    "odd_one_out": [2, 1, 2, 2, 2, 1],
    
    # General Awareness
    "defense": [5, 5, 4, 5, 4, 5],
    "current_affairs": [6, 7, 7, 6, 7, 7],
    "history": [3, 4, 3, 4, 3, 3],
    "geography": [3, 3, 4, 3, 3, 4],
    "polity": [2, 2, 2, 2, 2, 2],
    "science": [2, 2, 2, 3, 3, 2],
    "sports": [3, 3, 4, 3, 4, 3],
    "static_gk": [2, 2, 2, 2, 2, 2]
}

# =============================================================================
# DIFFICULTY DISTRIBUTION (Expected)
# =============================================================================
DIFFICULTY_DISTRIBUTION = {
    "easy": 0.40,      # 40 questions
    "medium": 0.40,    # 40 questions
    "hard": 0.20       # 20 questions
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    "topic_predictor": {
        "model_type": "xgboost",
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    "difficulty_classifier": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42
        }
    },
    "trend_analyzer": {
        "lookback_years": 5,
        "trend_threshold": 0.15  # 15% change considered significant
    }
}

# =============================================================================
# BIAS CORRECTION PARAMETERS
# =============================================================================
BIAS_CORRECTION = {
    "expected_easy_ratio": 0.45,
    "memory_recall_factor": 0.7,  # Memory-based papers capture ~70% of questions
    "difficulty_inflation": 1.3   # Hard questions over-reported by 30%
}

# =============================================================================
# CURRENT AFFAIRS CONFIGURATION
# =============================================================================
CURRENT_AFFAIRS_CONFIG = {
    "lookback_months": 6,
    "categories": [
        "Defense & Military",
        "International Relations",
        "Sports",
        "Awards & Honors",
        "Science & Technology",
        "Government Schemes",
        "Economy & Budget"
    ],
    "priority_keywords": [
        "indian air force", "iaf", "rafale", "tejas", "missile",
        "exercise", "operation", "bilateral", "summit", "treaty",
        "olympic", "asian games", "commonwealth", "world cup",
        "padma", "bharat ratna", "dronacharya", "arjuna award",
        "isro", "drdo", "space", "satellite", "nuclear",
        "budget", "gdp", "rbi", "fiscal"
    ]
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_CONFIG = {
    "confidence_threshold": 0.6,  # Only show predictions above 60% confidence
    "top_n_topics": 10,           # Show top 10 topics per section
    "generate_mock_blueprint": True,
    "export_formats": ["json", "csv", "html"]
}

# =============================================================================
# LOGGING
# =============================================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "logs" / f"afcat_prediction_{datetime.now().strftime('%Y%m%d')}.log"
}
