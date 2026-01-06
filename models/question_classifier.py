"""
Question Classifier for AFCAT Topics
=====================================
Classifies extracted questions into AFCAT topics using:
1. Zone Enforcement (question number ranges for 100-Q papers)
2. Gemini API classification (primary LLM method)
3. Zero-shot transformer classification (if available)
4. Keyword-based classification (fallback)
Also detects question types (calculation, factual, conceptual, etc.)
"""

import re
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import requests
except ImportError:
    requests = None

# Gemini API integration (New SDK - google.genai)

AFCAT_CLASSIFICATION = {
    # ==================== VERBAL ABILITY (30 Q) ====================
    "verbal_ability": {
        "section_range": (1, 30),
        "topics": {
            "VA_COMP": {
                "name": "Reading Comprehension",
                "keywords": ["passage", "paragraph", "according to the passage", "author", 
                            "based on the passage", "main idea", "central theme", "inferred",
                            "read the following passage", "title of the passage", "comprehension"],
                "weight": 3.0
            },
            "VA_CLOZE": {
                "name": "Cloze Test",
                "keywords": ["cloze", "fill in the blanks", "passage with blanks", "blanks numbered",
                            "suitable words", "choose the appropriate word"],
                "weight": 2.5
            },
            "VA_SYN": {
                "name": "Synonyms",
                "keywords": ["synonym", "similar meaning", "same meaning", "closest in meaning",
                            "means the same", "word similar", "nearly same meaning"],
                "weight": 2.5
            },
            "VA_ANT": {
                "name": "Antonyms", 
                "keywords": ["antonym", "opposite", "contrary", "opposite meaning",
                            "most opposite", "reverse", "contrary to"],
                "weight": 2.5
            },
            "VA_ERR": {
                "name": "Error Detection/Spotting",
                "keywords": ["error", "incorrect", "mistake", "wrong part", "no error",
                            "grammatical error", "find the error", "spot the error",
                            "underlined part", "grammatically correct", "identify the error"],
                "weight": 2.5
            },
            "VA_SEN_REAR": {
                "name": "Sentence Rearrangement",
                "keywords": ["rearrange", "jumbled", "proper order", "correct order",
                            "meaningful paragraph", "arrange the sentences", "logical sequence",
                            "para jumbles", "sentence order", "jumbled sentence"],
                "weight": 2.5
            },
            "VA_OWS": {
                "name": "One Word Substitution",
                "keywords": ["one word", "single word", "substitute", "expressed in one word",
                            "one word for", "substitution", "replaced by one word"],
                "weight": 2.5
            },
            "VA_IDIOM": {
                "name": "Idioms & Phrases",
                "keywords": ["idiom", "phrase", "proverb", "expression", "saying",
                            "metaphor", "figure of speech", "idiomatic expression",
                            "the phrase means", "meaning of the idiom", "phrase means"],
                "weight": 2.5
            },
            "VA_SEN_COMP": {
                "name": "Sentence Completion",
                "keywords": ["fill in the blank", "complete", "suitable word", "appropriate word",
                            "blank", "missing word", "choose the correct word"],
                "weight": 2.0
            },
            "VA_SEN_IMP": {
                "name": "Sentence Improvement",
                "keywords": ["improve", "improvement", "best replaces", "no improvement",
                            "replace the underlined", "improved by", "better alternative"],
                "weight": 2.5
            },
            "VA_VOICE": {
                "name": "Active-Passive Voice",
                "keywords": ["active", "passive", "voice", "change the voice",
                            "convert to passive", "convert to active"],
                "weight": 2.0
            },
            "VA_NARR": {
                "name": "Direct-Indirect Speech",
                "keywords": ["direct", "indirect", "speech", "reported speech",
                            "narration", "change into indirect", "indirect speech"],
                "weight": 2.0
            },
            "VA_SPELL": {
                "name": "Spelling",
                "keywords": ["spelling", "spelt", "correctly spelt", "misspelt",
                            "wrongly spelt", "correct spelling", "spelled correctly"],
                "weight": 2.0
            },
            "VA_VOCAB": {
                "name": "Vocabulary",
                "keywords": ["meaning", "vocabulary", "define", "word meaning",
                            "meaning of the word"],
                "weight": 1.5
            },
            "VA_ANAL": {
                "name": "Verbal Analogy",
                "keywords": ["analogy", "related", "is to", "as", "same way",
                            "similar relationship", "word analogy", "complete the analogy"],
                "weight": 2.0
            }
        }
    },
    
    # ==================== GENERAL AWARENESS (25 Q) ====================
    "general_awareness": {
        "section_range": (31, 55),
        "topics": {
            "GA_HIST_ANC": {
                "name": "Ancient History",
                "keywords": ["ancient", "harappa", "indus valley", "vedic", "maurya", "ashoka",
                            "gupta", "dynasty", "emperor", "buddha", "mahavira", "stone age",
                            "bronze age", "buddhism", "jainism", "chandragupta", "magadha"],
                "weight": 2.5
            },
            "GA_HIST_MED": {
                "name": "Medieval History",
                "keywords": ["medieval", "mughal", "delhi sultanate", "akbar", "aurangzeb",
                            "shah jahan", "chola", "vijayanagara", "rajput", "maratha",
                            "shivaji", "babur", "humayun", "sultanate", "khilji", "tughlaq"],
                "weight": 2.5
            },
            "GA_HIST_MOD": {
                "name": "Modern History",
                "keywords": ["modern", "british", "freedom", "independence", "gandhi", "nehru",
                            "revolt", "1857", "movement", "quit india", "civil disobedience",
                            "non-cooperation", "dandi march", "partition", "congress", "swadeshi"],
                "weight": 2.5
            },
            "GA_GEO_IND": {
                "name": "Indian Geography",
                "keywords": ["river", "mountain", "state", "capital", "boundary", "climate",
                            "soil", "forest", "himalaya", "ganges", "monsoon", "plateau",
                            "peninsula", "coastal", "western ghats", "eastern ghats", "deccan"],
                "weight": 2.5
            },
            "GA_GEO_WLD": {
                "name": "World Geography",
                "keywords": ["country", "ocean", "continent", "largest", "longest", "highest",
                            "world", "international", "pacific", "atlantic", "everest",
                            "amazon", "nile", "sahara", "equator", "latitude", "longitude"],
                "weight": 2.5
            },
            "GA_POLITY": {
                "name": "Polity & Constitution",
                "keywords": ["constitution", "article", "amendment", "parliament", "supreme court",
                            "president", "prime minister", "governor", "election", "fundamental rights",
                            "directive principles", "lok sabha", "rajya sabha", "judiciary",
                            "executive", "legislature", "preamble", "citizenship", "writ"],
                "weight": 2.5
            },
            "GA_ECON": {
                "name": "Economy & Finance",
                "keywords": ["gdp", "budget", "rbi", "bank", "inflation", "fiscal", "monetary",
                            "tax", "gst", "economy", "finance", "sebi", "stock", "niti aayog",
                            "planning commission", "reserve bank", "fiscal deficit", "revenue"],
                "weight": 2.5
            },
            "GA_SCI_PHY": {
                "name": "Physics",
                "keywords": ["newton", "physics", "force", "motion", "gravity", "energy",
                            "work", "power", "electricity", "magnetism", "light", "sound",
                            "wave", "einstein", "relativity", "thermodynamics", "optics"],
                "weight": 2.5
            },
            "GA_SCI_CHEM": {
                "name": "Chemistry",
                "keywords": ["element", "compound", "atom", "molecule", "periodic table",
                            "chemistry", "acid", "base", "salt", "metal", "non-metal",
                            "reaction", "chemical", "formula", "oxidation", "reduction"],
                "weight": 2.5
            },
            "GA_SCI_BIO": {
                "name": "Biology",
                "keywords": ["cell", "organ", "biology", "plant", "animal", "human body",
                            "disease", "vitamin", "protein", "dna", "genetics", "evolution",
                            "ecosystem", "photosynthesis", "blood", "heart", "brain"],
                "weight": 2.5
            },
            "GA_DEF": {
                "name": "Defense & Military",
                "keywords": ["army", "navy", "air force", "indian air force", "iaf", "missile",
                            "operation", "war", "chief", "regiment", "exercise", "aircraft",
                            "tank", "submarine", "helicopter", "fighter jet", "brigade",
                            "military", "weapon", "defence", "cds", "general", "marshal", "admiral"],
                "weight": 3.0  # Higher weight for AFCAT
            },
            "GA_CURR": {
                "name": "Current Affairs",
                "keywords": ["recently", "launched", "inaugurated", "summit", "agreement",
                            "appointed", "new", "first", "2024", "2025", "2026", "latest",
                            "current", "announced", "signed", "elected", "visit"],
                "weight": 2.0
            },
            "GA_SPORTS": {
                "name": "Sports",
                "keywords": ["cricket", "hockey", "football", "olympics", "trophy", "cup",
                            "champion", "world cup", "tournament", "player", "medal",
                            "gold", "silver", "bronze", "ipl", "fifa", "asian games",
                            "commonwealth games", "winner", "captain", "stadium"],
                "weight": 2.5
            },
            "GA_AWARD": {
                "name": "Awards & Honours",
                "keywords": ["award", "prize", "padma", "bharat ratna", "nobel", "dronacharya",
                            "arjuna", "sahitya", "national film award", "jnanpith",
                            "shanti swarup bhatnagar", "gallantry", "param vir chakra"],
                "weight": 2.5
            },
            "GA_BOOK": {
                "name": "Books & Authors",
                "keywords": ["book", "author", "written by", "wrote", "novel", "autobiography",
                            "biography", "published", "writer", "literature"],
                "weight": 2.0
            },
            "GA_DAYS": {
                "name": "Important Days",
                "keywords": ["day", "observed on", "celebrated", "international day",
                            "national day", "world day", "anniversary", "commemorate"],
                "weight": 2.0
            },
            "GA_ORG": {
                "name": "Organizations",
                "keywords": ["un", "united nations", "who", "imf", "world bank", "nato",
                            "asean", "saarc", "brics", "g20", "g7", "organization",
                            "headquarters", "secretary general", "unicef", "unesco"],
                "weight": 2.5
            },
            "GA_TECH": {
                "name": "Technology & Space",
                "keywords": ["technology", "computer", "software", "internet", "ai", "satellite",
                            "isro", "space", "digital", "cyber", "artificial intelligence",
                            "machine learning", "nasa", "rocket", "chandrayaan", "mangalyaan"],
                "weight": 2.5
            },
            "GA_CULT": {
                "name": "Art & Culture",
                "keywords": ["art", "culture", "dance", "music", "classical", "folk",
                            "festival", "heritage", "unesco", "bharatanatyam", "kathak",
                            "temple", "architecture", "painting", "sculpture"],
                "weight": 2.0
            }
        }
    },
    
    # ==================== REASONING (25 Q) ====================
    "reasoning": {
        "section_range": (56, 80),
        "topics": {
            # Verbal Reasoning Topics
            "RM_VR_ANA": {
                "name": "Analogy (Verbal)",
                "keywords": ["analogy", "related", "is to", "as", "same way",
                            "relationship", "pair", "similar relationship", "is related to"],
                "weight": 2.5
            },
            "RM_VR_CODE": {
                "name": "Coding-Decoding",
                "keywords": ["code", "coded", "decode", "written as", "cipher",
                            "coded as", "certain code", "in a certain code",
                            "coded language", "letter coding", "number coding"],
                "weight": 3.0
            },
            "RM_VR_BLOOD": {
                "name": "Blood Relations",
                "keywords": ["father", "mother", "brother", "sister", "son", "daughter",
                            "relation", "nephew", "niece", "uncle", "aunt", "cousin",
                            "grandfather", "grandmother", "husband", "wife", "in-law",
                            "how is related", "relationship between", "family"],
                "weight": 3.0
            },
            "RM_VR_DIR": {
                "name": "Direction Sense",
                "keywords": ["north", "south", "east", "west", "direction", "facing",
                            "turn", "left", "right", "walked", "displacement",
                            "starting point", "towards", "north-east", "south-west"],
                "weight": 2.5
            },
            "RM_VR_SYL": {
                "name": "Syllogism",
                "keywords": ["conclusion", "follows", "all", "some", "no", "statement",
                            "conclusions", "premises", "valid", "definitely true",
                            "only conclusion", "both conclusions", "neither conclusion"],
                "weight": 2.5
            },
            "RM_VR_VENN": {
                "name": "Venn Diagrams",
                "keywords": ["venn", "diagram", "represents", "relationship", "circles",
                            "best represents", "intersection", "venn diagram", "overlapping"],
                "weight": 2.5
            },
            "RM_VR_SER": {
                "name": "Series Completion (Reasoning)",
                "keywords": ["series", "next", "missing", "complete", "pattern",
                            "what comes next", "letter series", "complete the series",
                            "find the missing"],
                "weight": 2.5
            },
            "RM_VR_ODD": {
                "name": "Odd One Out/Classification",
                "keywords": ["odd", "different", "does not belong", "unlike",
                            "which is different", "exception", "odd one out",
                            "does not fit", "one is different", "classify", "classification"],
                "weight": 2.5
            },
            "RM_VR_RANK": {
                "name": "Ranking & Ordering",
                "keywords": ["rank", "position", "from top", "from bottom", "order",
                            "arrange", "sequence", "ranking", "position from left",
                            "position from right", "tallest", "shortest"],
                "weight": 2.5
            },
            "RM_VR_SEAT": {
                "name": "Seating Arrangement",
                "keywords": ["sitting", "seating", "arrangement", "row", "circle",
                            "facing", "between", "adjacent", "opposite",
                            "circular arrangement", "linear arrangement"],
                "weight": 2.5
            },
            "RM_VR_PUZ": {
                "name": "Puzzle",
                "keywords": ["puzzle", "based on information", "given information",
                            "read carefully", "conditions", "following information"],
                "weight": 2.0
            },
            "RM_VR_SUFF": {
                "name": "Data Sufficiency",
                "keywords": ["data sufficiency", "statement alone", "both statements",
                            "sufficient", "insufficient", "to answer the question",
                            "statement i alone", "statement ii alone"],
                "weight": 2.5
            },
            "RM_VR_LOG": {
                "name": "Logical Deduction",
                "keywords": ["deduce", "deduction", "logically follows", "must be true",
                            "if-then", "logical conclusion", "definitely true",
                            "statement", "assumption"],
                "weight": 2.5
            },
            "RM_VR_CLK": {
                "name": "Clocks",
                "keywords": ["clock", "minute", "hour", "angle", "hands", "time",
                            "overlap", "straight line", "angle between hands",
                            "minute hand", "hour hand", "what time"],
                "weight": 2.5
            },
            "RM_VR_CAL": {
                "name": "Calendar",
                "keywords": ["calendar", "day", "date", "week", "month", "year",
                            "what day", "leap year", "day of the week", "which day",
                            "january", "february", "odd days"],
                "weight": 2.5
            },
            # Non-Verbal/Visual Reasoning Topics
            "RM_NV_FIG": {
                "name": "Figure Series/Pattern",
                "keywords": ["figure", "pattern", "next figure", "missing figure",
                            "figure pattern", "complete the figure", "sequence of figures",
                            "following figures", "which figure"],
                "weight": 3.0
            },
            "RM_NV_SPA": {
                "name": "Spatial Ability",
                "keywords": ["rotate", "rotation", "3d", "fold", "unfold", "spatial",
                            "perspective", "three dimensional", "orientation",
                            "spatial orientation", "rotated"],
                "weight": 3.0
            },
            "RM_NV_EMB": {
                "name": "Embedded Figures",
                "keywords": ["embedded", "hidden", "contains", "which figure is embedded",
                            "hidden figure", "embedded in", "figure is hidden", "find the hidden"],
                "weight": 3.0
            },
            "RM_NV_MIR": {
                "name": "Mirror & Water Images",
                "keywords": ["mirror", "water", "image", "reflection", "mirror image",
                            "water image", "reflected", "mirror reflection"],
                "weight": 3.0
            },
            "RM_NV_PAP": {
                "name": "Paper Folding & Cutting",
                "keywords": ["paper", "fold", "punch", "cut", "unfold", "holes",
                            "folded", "paper is folded", "punched", "cutting"],
                "weight": 3.0
            },
            "RM_NV_CUBE": {
                "name": "Cubes & Dice",
                "keywords": ["cube", "dice", "painted", "faces", "opposite face",
                            "adjacent", "sides of dice", "opposite side", "unfolded cube"],
                "weight": 3.0
            },
            "RM_NV_MAT": {
                "name": "Matrix/Grid Patterns",
                "keywords": ["matrix", "grid", "box", "missing element", "complete the matrix",
                            "3x3", "find the missing", "select the figure"],
                "weight": 2.5
            }
        }
    },
    
    # ==================== NUMERICAL ABILITY (20 Q) ====================
    "numerical_ability": {
        "section_range": (81, 100),
        "topics": {
            "NA_PER": {
                "name": "Percentages",
                "keywords": ["percent", "%", "increase", "decrease", "percentage of",
                            "what percent", "reduced by", "increased by", "of the total",
                            "percentage change", "percentage increase", "percentage decrease"],
                "weight": 2.5
            },
            "NA_RAT": {
                "name": "Ratio & Proportion",
                "keywords": ["ratio", "proportion", "divided", "share", "parts",
                            "distributed", "in the ratio", "a:b", "proportional",
                            "directly proportional", "inversely proportional"],
                "weight": 2.5
            },
            "NA_PL": {
                "name": "Profit & Loss",
                "keywords": ["profit", "loss", "cost price", "selling price", "cp", "sp",
                            "discount", "marked price", "gain", "sold", "bought",
                            "merchant", "successive discount", "markup", "shopkeeper", "trader"],
                "weight": 2.5
            },
            "NA_SI_CI": {
                "name": "Simple & Compound Interest",
                "keywords": ["interest", "principal", "rate", "annum", "compound",
                            "simple interest", "si", "ci", "years", "amount",
                            "compounded", "annually", "half-yearly", "quarterly",
                            "rate of interest", "per annum", "p.a."],
                "weight": 2.5
            },
            "NA_AVG": {
                "name": "Averages",
                "keywords": ["average", "mean", "sum", "total", "average age",
                            "average weight", "new average", "overall average",
                            "average of", "find the average", "weighted average"],
                "weight": 2.5
            },
            "NA_TW": {
                "name": "Time & Work",
                "keywords": ["work", "days", "complete", "efficiency", "together", "alone",
                            "men", "women", "finish", "remaining", "pipe", "fill", "empty",
                            "cistern", "tank", "hours to complete", "working together",
                            "can do a work", "wages", "contractor"],
                "weight": 2.5
            },
            "NA_STD": {
                "name": "Speed, Time & Distance",
                "keywords": ["speed", "km/hr", "m/s", "distance", "travel", "relative speed",
                            "overtake", "average speed", "running", "walking", "cycling",
                            "time taken", "covered", "journey"],
                "weight": 2.5
            },
            "NA_TRAIN": {
                "name": "Trains & Boats",
                "keywords": ["train", "platform", "tunnel", "bridge", "pole", "crosses",
                            "passes", "boat", "stream", "upstream", "downstream",
                            "still water", "current", "speed of stream", "length of train"],
                "weight": 2.5
            },
            "NA_FRAC": {
                "name": "Decimal & Fraction",
                "keywords": ["decimal", "fraction", "numerator", "denominator",
                            "proper fraction", "improper fraction", "mixed fraction",
                            "convert to decimal", "recurring decimal", "simplify fraction"],
                "weight": 2.0
            },
            "NA_SER": {
                "name": "Number Series",
                "keywords": ["series", "next number", "pattern", "sequence", "missing number",
                            "wrong number", "find the next", "complete the series",
                            "what comes next", "number pattern"],
                "weight": 2.5
            },
            "NA_SIMP": {
                "name": "Simplification",
                "keywords": ["simplify", "evaluate", "value of", "solve", "calculate",
                            "find the value", "expression", "bodmas", "order of operations"],
                "weight": 2.0
            },
            "NA_ALG": {
                "name": "Algebra",
                "keywords": ["equation", "solve for", "x", "y", "variable", "quadratic",
                            "linear", "polynomial", "root", "if x =", "find x",
                            "algebraic expression", "factorize", "expand"],
                "weight": 2.5
            },
            "NA_GEO": {
                "name": "Geometry",
                "keywords": ["angle", "triangle", "circle", "parallel", "perpendicular",
                            "degree", "bisector", "chord", "tangent", "radius", "diameter",
                            "arc", "inscribed", "circumscribed", "polygon", "acute", "obtuse",
                            "equilateral", "isosceles", "quadrilateral"],
                "weight": 2.5
            },
            "NA_MENS": {
                "name": "Mensuration",
                "keywords": ["area", "volume", "perimeter", "surface area", "cube", "cylinder",
                            "cone", "sphere", "rectangle", "square", "hemisphere",
                            "curved surface", "lateral surface", "diagonal", "circumference"],
                "weight": 2.5
            },
            "NA_PROB": {
                "name": "Probability",
                "keywords": ["probability", "dice", "cards", "ball", "bag", "drawn",
                            "random", "chance", "likely", "odds", "favorable",
                            "outcomes", "fair dice", "deck of cards", "without replacement"],
                "weight": 2.5
            },
            "NA_PNC": {
                "name": "Permutation & Combination",
                "keywords": ["permutation", "combination", "ways", "arrangements",
                            "selection", "choose", "arrange", "n!", "factorial",
                            "how many ways", "can be arranged", "can be selected"],
                "weight": 2.5
            },
            "NA_DI": {
                "name": "Data Interpretation",
                "keywords": ["table", "chart", "graph", "bar graph", "pie chart", "line graph",
                            "percentage increase", "data", "according to the table",
                            "study the graph", "following table", "given data"],
                "weight": 2.5
            },
            "NA_NUM": {
                "name": "Number System",
                "keywords": ["divisible", "factor", "multiple", "prime", "composite",
                            "remainder", "digit", "unit digit", "face value",
                            "place value", "divisibility", "prime number", "even", "odd"],
                "weight": 2.5
            },
            "NA_LCM": {
                "name": "LCM & HCF",
                "keywords": ["lcm", "hcf", "gcd", "least common multiple", "highest common factor",
                            "greatest common divisor", "common multiple", "common factor"],
                "weight": 2.5
            },
            "NA_MIX": {
                "name": "Mixtures & Alligation",
                "keywords": ["mixture", "alligation", "mixed", "concentration", "pure",
                            "water", "milk", "solution", "ratio of mixing",
                            "wine and water", "two types"],
                "weight": 2.5
            },
            "NA_CLK": {
                "name": "Clocks & Calendars",
                "keywords": ["clock", "minute", "hour", "hands of clock", "angle between hands",
                            "calendar", "day", "date", "leap year", "ordinary year",
                            "what day", "day of the week"],
                "weight": 2.0
            }
        }
    }
}

# Non-verbal reasoning detection patterns (for figure-based questions)
NON_VERBAL_PATTERNS = [
    r"figure\s*\d*",
    r"which\s+figure",
    r"next\s+figure",
    r"missing\s+figure",
    r"pattern\s+completion",
    r"complete\s+the\s+pattern",
    r"embedded\s+figure",
    r"hidden\s+figure",
    r"mirror\s+image",
    r"water\s+image",
    r"paper\s+folding",
    r"paper\s+cutting",
    r"cube|dice",
    r"rotation|rotated",
    r"spatial",
    r"matrix",
    r"grid"
]

# Default AFCAT section zones (used as hints, not rigid when zone_mode="flex"/"off")
SECTION_ZONES = {
    "verbal_ability": (1, 30),
    "general_awareness": (31, 55),
    "reasoning": (56, 80),
    "numerical_ability": (81, 100),
}

# Module logger
logger = logging.getLogger(__name__)

# Zone handling modes
# - strict: force section from question number (legacy behavior)
# - flex: only override when classifier confidence is low
# - off: never enforce zones
DEFAULT_ZONE_MODE = "flex"
STRICT_ZONE_ENFORCEMENT = False  # Backward compatibility flag (use zone_mode instead)
ZONE_OVERRIDE_THRESHOLD = 0.90

# Gemini API Key (set via environment or directly)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBskxPjCP0zg_hDj09qRSGu94PwgW4JGJM")

# Groq API Key (fast inference)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class ClassificationMethod(Enum):
    TRANSFORMER = "zero-shot-transformer"
    KEYWORD = "keyword-matching"
    HYBRID = "hybrid"
    ZONE_ENFORCED = "zone-enforced"
    GEMINI = "gemini-api"


@dataclass
class ClassificationResult:
    """Result of topic classification."""
    section: str
    topic: str
    confidence: float
    subtopic: Optional[str] = None
    question_type: Optional[str] = None
    method: str = "unknown"
    alternate_topics: Optional[List[Tuple[str, float]]] = None


class AFCATTopicClassifier:
    """
    Multi-method topic classifier for AFCAT questions.
    Uses zero-shot transformers when available, with keyword fallback.
    """
    
    # Complete AFCAT 2026 topic taxonomy with comprehensive keywords from official syllabus
    TOPIC_TAXONOMY = {
        "numerical_ability": {
            "topics": [
                "decimal_fraction", "time_and_work", "profit_loss",
                "percentages", "ratio_proportion", "simple_compound_interest",
                "averages", "number_series", "simplification", "algebra",
                "geometry", "mensuration", "probability", "permutation_combination",
                "data_interpretation", "number_system", "lcm_hcf", "mixtures_alligation",
                "speed_time_distance", "trains_boats", "clocks_calendars"
            ],
            "keywords": {
                "decimal_fraction": [
                    "decimal", "fraction", "numerator", "denominator", "proper fraction",
                    "improper fraction", "mixed fraction", "convert to decimal", "recurring decimal",
                    "terminating decimal", "simplify fraction", "equivalent fraction"
                ],
                "speed_time_distance": [
                    "speed", "train", "km/hr", "m/s", "distance", "travel", 
                    "upstream", "downstream", "boat", "stream", "current",
                    "relative speed", "overtake", "cross", "platform", "tunnel",
                    "average speed", "running", "walking", "cycling", "flight"
                ],
                "trains_boats": [
                    "train", "platform", "tunnel", "bridge", "pole", "crosses",
                    "passes", "boat", "stream", "upstream", "downstream", "still water",
                    "current", "speed of stream", "speed of boat"
                ],
                "time_and_work": [
                    "work", "days", "complete", "efficiency", "together", "alone",
                    "men", "women", "finish", "remaining", "pipe", "fill", "empty",
                    "cistern", "tank", "hours to complete", "working together",
                    "can do a work", "wages", "contractor"
                ],
                "profit_loss": [
                    "profit", "loss", "cost price", "selling price", "cp", "sp",
                    "discount", "marked price", "gain", "sold", "bought", "merchant",
                    "successive discount", "markup", "shopkeeper", "trader"
                ],
                "percentages": [
                    "percent", "%", "increase", "decrease", "percentage of",
                    "what percent", "reduced by", "increased by", "of the total",
                    "percentage change", "percentage increase", "percentage decrease"
                ],
                "ratio_proportion": [
                    "ratio", "proportion", "divided", "share", "parts",
                    "distributed", "in the ratio", "a:b", "mixture", "proportional",
                    "directly proportional", "inversely proportional"
                ],
                "simple_compound_interest": [
                    "interest", "principal", "rate", "annum", "compound",
                    "simple interest", "si", "ci", "years", "amount",
                    "compounded", "annually", "half-yearly", "quarterly",
                    "rate of interest", "per annum", "p.a."
                ],
                "averages": [
                    "average", "mean", "sum", "total", "average age",
                    "average weight", "new average", "overall average",
                    "average of", "find the average", "weighted average"
                ],
                "number_series": [
                    "series", "next number", "pattern", "sequence", "missing number",
                    "wrong number", "find the next", "complete the series",
                    "what comes next", "number pattern", "find the missing"
                ],
                "simplification": [
                    "simplify", "evaluate", "value of", "solve", "calculate",
                    "find the value", "expression", "bodmas", "order of operations"
                ],
                "algebra": [
                    "equation", "solve for", "x", "y", "variable", "quadratic",
                    "linear", "polynomial", "root", "if x =", "find x",
                    "algebraic expression", "factorize", "expand"
                ],
                "geometry": [
                    "angle", "triangle", "circle", "parallel", "perpendicular",
                    "degree", "bisector", "chord", "tangent", "radius",
                    "diameter", "arc", "inscribed", "circumscribed", "polygon",
                    "acute", "obtuse", "right angle", "equilateral", "isosceles"
                ],
                "mensuration": [
                    "area", "volume", "perimeter", "surface area", "cube", "cylinder",
                    "cone", "sphere", "rectangle", "square", "hemisphere",
                    "curved surface", "lateral surface", "diagonal", "circumference"
                ],
                "probability": [
                    "probability", "dice", "cards", "ball", "bag", "drawn",
                    "random", "chance", "likely", "odds", "favorable",
                    "outcomes", "fair dice", "deck of cards", "without replacement"
                ],
                "permutation_combination": [
                    "permutation", "combination", "ways", "arrangements",
                    "selection", "choose", "arrange", "n!", "factorial",
                    "how many ways", "can be arranged", "can be selected"
                ],
                "data_interpretation": [
                    "table", "chart", "graph", "bar graph", "pie chart", "line graph",
                    "percentage increase", "data", "according to the table",
                    "study the graph", "following table", "given data"
                ],
                "number_system": [
                    "divisible", "factor", "multiple", "prime", "composite",
                    "remainder", "digit", "unit digit", "face value",
                    "place value", "divisibility", "prime number", "even", "odd"
                ],
                "lcm_hcf": [
                    "lcm", "hcf", "gcd", "least common multiple", "highest common factor",
                    "greatest common divisor", "common multiple", "common factor"
                ],
                "mixtures_alligation": [
                    "mixture", "alligation", "mixed", "concentration",
                    "pure", "water", "milk", "solution", "ratio of mixing",
                    "wine and water", "two types"
                ],
                "clocks_calendars": [
                    "clock", "minute", "hour", "hands of clock", "angle between hands",
                    "calendar", "day", "date", "leap year", "ordinary year",
                    "what day", "day of the week"
                ]
            }
        },
        "verbal_ability": {
            "topics": [
                "reading_comprehension", "cloze_test", "synonyms", "antonyms",
                "error_detection", "sentence_rearrangement", "one_word_substitution",
                "idioms_phrases", "analogy_verbal", "sentence_completion",
                "para_jumbles", "active_passive", "direct_indirect",
                "spelling", "vocabulary", "sentence_improvement"
            ],
            "keywords": {
                "reading_comprehension": [
                    "passage", "according to", "author", "paragraph",
                    "read the passage", "comprehension", "based on the passage",
                    "what does the author", "main idea", "central theme",
                    "inferred from", "title of the passage", "according to passage",
                    "in the passage", "read the following"
                ],
                "cloze_test": [
                    "cloze", "fill in the blanks", "passage with blanks",
                    "suitable words", "given passage", "blanks numbered",
                    "choose the appropriate word"
                ],
                "synonyms": [
                    "synonym", "similar meaning", "same meaning", "most similar",
                    "closest in meaning", "means the same", "similar to",
                    "word similar", "nearly same meaning"
                ],
                "antonyms": [
                    "antonym", "opposite", "contrary", "opposite meaning",
                    "most opposite", "reverse", "opposite of", "contrary to"
                ],
                "error_detection": [
                    "error", "incorrect", "mistake", "wrong part", "no error",
                    "grammatical error", "find the error", "underlined part",
                    "spot the error", "which part has error", "grammatically correct",
                    "identify the error"
                ],
                "sentence_rearrangement": [
                    "rearrange", "jumbled", "proper order", "sequence",
                    "correct order", "sentences", "paragraph", "meaningful paragraph",
                    "arrange the sentences", "logical sequence", "sentence order"
                ],
                "one_word_substitution": [
                    "one word", "single word", "substitute", "word for",
                    "expressed in one word", "one word for", "substitution",
                    "replaced by one word"
                ],
                "idioms_phrases": [
                    "idiom", "phrase", "meaning of", "proverb", "expression",
                    "saying", "metaphor", "figure of speech", "idiomatic expression",
                    "the phrase means", "meaning of the idiom"
                ],
                "analogy_verbal": [
                    "related", "same way", "is to", "as", "analogy",
                    "relationship", "pair", "similar relationship",
                    "word analogy", "complete the analogy"
                ],
                "sentence_completion": [
                    "fill in the blank", "complete", "suitable word",
                    "appropriate word", "blank", "missing word",
                    "choose the correct word"
                ],
                "para_jumbles": [
                    "jumbled", "rearrange", "proper order", "sequence",
                    "correct order", "paragraph", "meaningful order"
                ],
                "active_passive": [
                    "active", "passive", "voice", "convert", "change the voice",
                    "passive voice", "active voice", "change into passive"
                ],
                "direct_indirect": [
                    "direct", "indirect", "speech", "reported speech",
                    "narration", "said", "told", "asked", "change into indirect",
                    "indirect speech"
                ],
                "spelling": [
                    "spelling", "spelt", "correctly spelt", "misspelt",
                    "wrongly spelt", "correct spelling", "spelled correctly"
                ],
                "vocabulary": [
                    "meaning", "word", "vocabulary", "define", "word meaning"
                ],
                "sentence_improvement": [
                    "improve", "improvement", "best replaces", "underlined part",
                    "no improvement", "replace the underlined", "improved by"
                ]
            }
        },
        "reasoning": {
            "topics": [
                "coding_decoding", "analogy", "blood_relations", "direction_sense",
                "syllogism", "venn_diagrams", "series_completion", "odd_one_out",
                "spatial_ability", "embedded_figures", "mirror_water_images",
                "paper_folding", "cubes_dice", "calendar", "clocks",
                "ranking_ordering", "seating_arrangement", "puzzle",
                "data_sufficiency", "pattern_completion", "logical_deduction",
                "classification"
            ],
            "keywords": {
                "coding_decoding": [
                    "code", "coded", "decode", "if", "then", "written as",
                    "language", "cipher", "coded as", "certain code",
                    "in a certain code", "coded language", "letter coding"
                ],
                "analogy": [
                    "related", "same way", "is to", "as", "analogy",
                    "relationship", "pair", "similar relationship",
                    "is related to", "in the same way"
                ],
                "blood_relations": [
                    "father", "mother", "brother", "sister", "son", "daughter",
                    "relation", "nephew", "niece", "uncle", "aunt", "cousin",
                    "grandfather", "grandmother", "husband", "wife", "in-law",
                    "how is related", "relationship between"
                ],
                "direction_sense": [
                    "north", "south", "east", "west", "direction", "facing",
                    "turn", "left", "right", "walked", "distance", "displacement",
                    "starting point", "towards", "north-east", "south-west"
                ],
                "syllogism": [
                    "conclusion", "follows", "all", "some", "no", "statement",
                    "conclusions", "premises", "valid", "definitely true",
                    "only conclusion", "both conclusions", "neither conclusion"
                ],
                "venn_diagrams": [
                    "venn", "diagram", "represents", "relationship", "circles",
                    "best represents", "intersection", "venn diagram"
                ],
                "series_completion": [
                    "series", "next", "missing", "complete", "pattern",
                    "what comes next", "figure series", "number series",
                    "letter series", "complete the series"
                ],
                "odd_one_out": [
                    "odd", "different", "does not belong", "unlike",
                    "which is different", "exception", "odd one out",
                    "does not fit", "one is different"
                ],
                "spatial_ability": [
                    "rotate", "rotation", "3d", "fold", "unfold",
                    "spatial", "perspective", "three dimensional",
                    "orientation", "spatial orientation"
                ],
                "embedded_figures": [
                    "embedded", "hidden", "figure", "contains",
                    "which figure is embedded", "hidden figure",
                    "embedded in", "figure is hidden"
                ],
                "mirror_water_images": [
                    "mirror", "water", "image", "reflection",
                    "mirror image", "water image", "reflected"
                ],
                "paper_folding": [
                    "paper", "fold", "punch", "cut", "unfold",
                    "holes", "folded", "paper is folded", "punched"
                ],
                "cubes_dice": [
                    "cube", "dice", "painted", "faces", "opposite face",
                    "adjacent", "sides of dice", "opposite side"
                ],
                "calendar": [
                    "calendar", "day", "date", "week", "month", "year",
                    "what day", "leap year", "january", "february",
                    "day of the week", "which day"
                ],
                "clocks": [
                    "clock", "minute", "hour", "angle", "hands",
                    "time", "between", "overlap", "straight line",
                    "angle between hands", "minute hand", "hour hand"
                ],
                "ranking_ordering": [
                    "rank", "position", "from top", "from bottom",
                    "order", "arrange", "sequence", "ranking",
                    "position from left", "position from right"
                ],
                "seating_arrangement": [
                    "sitting", "seating", "arrangement", "row", "circle",
                    "facing", "between", "adjacent", "opposite",
                    "circular arrangement", "linear arrangement"
                ],
                "puzzle": [
                    "puzzle", "based on information", "given information",
                    "read carefully", "conditions", "following information"
                ],
                "data_sufficiency": [
                    "data sufficiency", "statement alone", "both statements",
                    "sufficient", "insufficient", "to answer the question",
                    "statement i alone", "statement ii alone"
                ],
                "pattern_completion": [
                    "pattern", "complete the pattern", "next figure",
                    "missing figure", "figure pattern", "complete the figure"
                ],
                "logical_deduction": [
                    "deduce", "deduction", "logically follows", "must be true",
                    "if-then", "logical conclusion", "definitely true"
                ],
                "classification": [
                    "classify", "classification", "group", "category",
                    "does not belong to group", "odd one", "different from others"
                ]
            }
        },
        "general_awareness": {
            "topics": [
                "defense", "history_ancient", "history_medieval", "history_modern",
                "geography_india", "geography_world", "polity", "economy",
                "science_physics", "science_chemistry", "science_biology",
                "current_affairs", "sports", "awards",
                "books_authors", "important_days", "environment",
                "technology", "art_culture", "organizations", "personalities"
            ],
            "keywords": {
                "defense": [
                    "army", "navy", "air force", "indian air force", "iaf",
                    "missile", "operation", "war", "chief", "regiment",
                    "exercise", "aircraft", "tank", "submarine", "helicopter",
                    "fighter jet", "brigade", "military", "weapon", "defence",
                    "cds", "chief of defence staff", "general", "marshal",
                    "admiral", "squadron", "battalion"
                ],
                "history_ancient": [
                    "ancient", "harappa", "indus valley", "vedic", "maurya",
                    "ashoka", "gupta", "dynasty", "emperor", "buddha", "mahavira",
                    "stone age", "bronze age", "buddhism", "jainism", "chandragupta"
                ],
                "history_medieval": [
                    "medieval", "mughal", "delhi sultanate", "akbar", "aurangzeb",
                    "shah jahan", "chola", "vijayanagara", "rajput", "maratha",
                    "shivaji", "babur", "humayun", "sultanate"
                ],
                "history_modern": [
                    "modern", "british", "freedom", "independence", "gandhi",
                    "nehru", "revolt", "1857", "movement", "quit india",
                    "civil disobedience", "non-cooperation", "dandi march",
                    "partition", "congress", "independence movement"
                ],
                "geography_india": [
                    "river", "mountain", "state", "capital", "boundary",
                    "climate", "soil", "forest", "himalaya", "ganges",
                    "monsoon", "plateau", "peninsula", "coastal",
                    "western ghats", "eastern ghats", "deccan"
                ],
                "geography_world": [
                    "country", "ocean", "continent", "largest", "longest",
                    "highest", "world", "international", "pacific", "atlantic",
                    "everest", "amazon", "nile", "sahara"
                ],
                "polity": [
                    "constitution", "article", "amendment", "parliament",
                    "supreme court", "president", "prime minister", "governor",
                    "election", "fundamental rights", "directive principles",
                    "lok sabha", "rajya sabha", "judiciary", "executive",
                    "legislature", "preamble", "citizenship"
                ],
                "economy": [
                    "gdp", "budget", "rbi", "bank", "inflation", "fiscal",
                    "monetary", "tax", "gst", "economy", "finance", "sebi",
                    "stock", "niti aayog", "five year plan", "planning commission",
                    "reserve bank", "fiscal deficit", "revenue"
                ],
                "science_physics": [
                    "newton", "physics", "force", "motion", "gravity",
                    "energy", "work", "power", "electricity", "magnetism",
                    "light", "sound", "wave", "einstein", "relativity"
                ],
                "science_chemistry": [
                    "element", "compound", "atom", "molecule", "periodic table",
                    "chemistry", "acid", "base", "salt", "metal", "non-metal",
                    "reaction", "chemical", "formula"
                ],
                "science_biology": [
                    "cell", "organ", "biology", "plant", "animal", "human body",
                    "disease", "vitamin", "protein", "dna", "genetics",
                    "evolution", "ecosystem", "photosynthesis"
                ],
                "current_affairs": [
                    "recently", "launched", "inaugurated", "summit", "agreement",
                    "appointed", "new", "first", "2024", "2025", "2026",
                    "latest", "current", "announced", "signed"
                ],
                "sports": [
                    "cricket", "hockey", "football", "olympics", "trophy", "cup",
                    "champion", "world cup", "tournament", "player", "medal",
                    "gold", "silver", "bronze", "ipl", "fifa", "asian games",
                    "commonwealth games", "winner", "captain"
                ],
                "awards": [
                    "award", "prize", "padma", "bharat ratna", "nobel",
                    "dronacharya", "arjuna", "sahitya", "national film award",
                    "shanti swarup bhatnagar", "jnanpith"
                ],
                "books_authors": [
                    "book", "author", "written by", "wrote", "novel",
                    "autobiography", "biography", "published", "writer"
                ],
                "important_days": [
                    "day", "observed on", "celebrated", "international day",
                    "national day", "world day", "anniversary", "commemorate"
                ],
                "environment": [
                    "environment", "pollution", "climate change", "biodiversity",
                    "ecosystem", "wildlife", "conservation", "forest",
                    "global warming", "greenhouse", "ozone"
                ],
                "technology": [
                    "technology", "computer", "software", "internet", "ai",
                    "satellite", "isro", "space", "digital", "cyber",
                    "artificial intelligence", "machine learning", "nasa"
                ],
                "art_culture": [
                    "art", "culture", "dance", "music", "classical",
                    "folk", "festival", "heritage", "unesco", "bharatanatyam",
                    "kathak", "temple", "architecture", "painting"
                ],
                "organizations": [
                    "un", "united nations", "who", "imf", "world bank",
                    "nato", "asean", "saarc", "brics", "g20", "g7",
                    "organization", "headquarters", "secretary general"
                ],
                "personalities": [
                    "founder", "invented", "discovered", "born", "died",
                    "famous", "known for", "biography", "personality",
                    "scientist", "leader", "freedom fighter"
                ]
            }
        }
    }
    
    # Human-readable topic names (updated for 2026 syllabus)
    TOPIC_LABELS = {
        # Numerical Ability
        "decimal_fraction": "Decimal & Fraction",
        "speed_time_distance": "Speed, Time & Distance",
        "trains_boats": "Trains & Boats",
        "time_and_work": "Time & Work",
        "profit_loss": "Profit & Loss",
        "percentages": "Percentages",
        "ratio_proportion": "Ratio & Proportion",
        "simple_compound_interest": "Simple & Compound Interest",
        "averages": "Averages",
        "number_series": "Number Series",
        "simplification": "Simplification",
        "algebra": "Algebra",
        "geometry": "Geometry",
        "mensuration": "Mensuration",
        "probability": "Probability",
        "permutation_combination": "Permutation & Combination",
        "data_interpretation": "Data Interpretation",
        "number_system": "Number System",
        "lcm_hcf": "LCM & HCF",
        "mixtures_alligation": "Mixtures & Alligation",
        "clocks_calendars": "Clocks & Calendars",
        
        # Verbal Ability
        "reading_comprehension": "Reading Comprehension",
        "cloze_test": "Cloze Test",
        "synonyms": "Synonyms",
        "antonyms": "Antonyms",
        "error_detection": "Error Detection",
        "sentence_rearrangement": "Sentence Rearrangement",
        "one_word_substitution": "One Word Substitution",
        "idioms_phrases": "Idioms & Phrases",
        "analogy_verbal": "Verbal Analogy",
        "sentence_completion": "Sentence Completion",
        "para_jumbles": "Para Jumbles",
        "active_passive": "Active-Passive Voice",
        "direct_indirect": "Direct-Indirect Speech",
        "spelling": "Spelling",
        "vocabulary": "Vocabulary",
        "sentence_improvement": "Sentence Improvement",
        
        # Reasoning
        "coding_decoding": "Coding-Decoding",
        "analogy": "Analogy",
        "blood_relations": "Blood Relations",
        "direction_sense": "Direction Sense",
        "syllogism": "Syllogism",
        "venn_diagrams": "Venn Diagrams",
        "series_completion": "Series Completion",
        "odd_one_out": "Odd One Out",
        "spatial_ability": "Spatial Ability",
        "embedded_figures": "Embedded Figures",
        "mirror_water_images": "Mirror & Water Images",
        "paper_folding": "Paper Folding",
        "cubes_dice": "Cubes & Dice",
        "calendar": "Calendar",
        "clocks": "Clocks",
        "ranking_ordering": "Ranking & Ordering",
        "seating_arrangement": "Seating Arrangement",
        "puzzle": "Puzzle",
        "data_sufficiency": "Data Sufficiency",
        "pattern_completion": "Pattern Completion",
        "logical_deduction": "Logical Deduction",
        "classification": "Classification",
        
        # General Awareness
        "defense": "Defense & Military",
        "history_ancient": "Ancient History",
        "history_medieval": "Medieval History",
        "history_modern": "Modern History",
        "geography_india": "Indian Geography",
        "geography_world": "World Geography",
        "polity": "Polity & Constitution",
        "economy": "Economy & Finance",
        "science_physics": "Physics",
        "science_chemistry": "Chemistry",
        "science_biology": "Biology",
        "current_affairs": "Current Affairs",
        "sports": "Sports",
        "awards": "Awards & Honours",
        "books_authors": "Books & Authors",
        "important_days": "Important Days",
        "environment": "Environment",
        "technology": "Technology",
        "art_culture": "Art & Culture",
        "organizations": "Organizations",
        "personalities": "Personalities",
        
        # Legacy mappings for backward compatibility
        "history": "History",
        "geography": "Geography",
        "science": "Science"
    }
    
    def __init__(
        self,
        use_transformers: bool = False,  # Default to keyword for faster setup
        transformer_model: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        use_ollama_fallback: bool = False,  # Disabled by default
        ollama_model: str = "llama3",
        use_gemini: bool = False,  # DISABLED for batch - use keywords+zones instead
        gemini_api_key: Optional[str] = None,
        use_groq: bool = True,  # Enable Groq API for better topic detection
        groq_api_key: Optional[str] = None,
        total_questions: int = 100,  # For zone enforcement decision
        zone_mode: str = DEFAULT_ZONE_MODE,  # "strict", "flex", "off"
    ):
        self.use_transformers = use_transformers
        self.transformer_model = transformer_model
        self.device = device
        self.use_ollama_fallback = use_ollama_fallback
        self.ollama_model = ollama_model
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        self.total_questions = total_questions
        self.zone_mode = zone_mode
        self._classifier = None
        self.use_gemini = use_gemini and gemini_api_key is not None
        self._gemini_client = None  # New SDK uses Client pattern
        self._gemini_model_name = "gemini-2.0-flash"
        self._gemini_call_count = 0
        self._last_gemini_call = 0
        
        # Initialize Gemini if available
        if self.use_gemini and self.gemini_api_key:
            self._init_gemini()
        
        # Groq API settings
        self.use_groq = use_groq
        self.groq_api_key = groq_api_key or GROQ_API_KEY
        self._last_groq_call = 0
        self._groq_call_count = 0
        
        if use_transformers:
            self._init_transformer()
    
    def _init_gemini(self):
        """Initialize Gemini API using new google.genai SDK."""
        try:
            # New SDK uses Client pattern
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            self._gemini_model_name = "gemini-2.0-flash"
            logger.info("Initialized Gemini 2.0 Flash API (new SDK)")
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")
            self.use_gemini = False
            self._gemini_client = None
            
    def _init_transformer(self):
        """Initialize zero-shot classifier."""
        try:
            from transformers import pipeline
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.transformer_model,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Loaded transformer: {self.transformer_model}")
        except Exception as e:
            logger.warning(f"Transformer init failed: {e}. Using keywords only.")
            self.use_transformers = False
            
    def classify(
        self,
        question_text: str,
        question_number: Optional[int] = None,
        section_hint: Optional[str] = None,
        use_llm_fallback: Optional[bool] = None,
        question_type: Optional[str] = None,
        single_section: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a question into AFCAT topics.
        
        Args:
            question_text: The question text to classify
            question_number: Question number for zone enforcement (100-Q papers only)
            section_hint: Optional section to narrow down topics
            use_llm_fallback: Override instance setting for LLM fallback
            question_type: Optional type hint from OCR (e.g., 'verbal_inferred', 'math_inferred')
            
        Returns:
            ClassificationResult with section, topic, confidence, etc.
        """
        target_section = single_section or self.single_section_override
        if target_section:
            target_section = target_section.strip().lower()
            if self.total_questions < 80:
                section_hint = target_section

        # SAFE MODE: Handle zone-based placeholder questions from OCR
        # These are gap-filled questions where we know the zone but NOT the topic
        # FIX: non_verbal_figure should ALSO respect question number zones
        # Non-verbal figures only appear in Q56-Q80 (Reasoning), but OCR may misidentify
        # other questions as figures due to "[LIKELY FIGURE]" markers
        if question_type in ("verbal_inferred", "ga_inferred", "math_inferred", "non_verbal_figure", "math_rescued", "inferred_gap", "unknown_gap"):
            # ZONE ENFORCEMENT: Use question_number to determine section for ALL placeholder types
            if question_number:
                if 1 <= question_number <= 30:
                    section, topic = "verbal_ability", "Inferred Verbal Ability"
                elif 31 <= question_number <= 55:
                    section, topic = "general_awareness", "Inferred General Awareness"
                elif 56 <= question_number <= 80:
                    section, topic = "reasoning", "Inferred Reasoning"
                elif 81 <= question_number <= 100:
                    section, topic = "numerical_ability", "Inferred Numerical Ability"
                else:
                    # Fallback to type-based inference for questions > 100
                    zone_sections = {
                        "verbal_inferred": ("verbal_ability", "Inferred Verbal Ability"),
                        "ga_inferred": ("general_awareness", "Inferred General Awareness"),
                        "math_inferred": ("numerical_ability", "Inferred Numerical Ability"),
                        "math_rescued": ("numerical_ability", "Inferred Numerical Ability"),
                        "non_verbal_figure": ("reasoning", "Inferred Reasoning"),
                        "inferred_gap": ("unknown", "Inferred Gap"),
                        "unknown_gap": ("unknown", "Unknown Gap"),
                    }
                    section, topic = zone_sections.get(question_type, ("unknown", "unknown"))
            else:
                # No question number available - use type-based inference
                zone_sections = {
                    "verbal_inferred": ("verbal_ability", "Inferred Verbal Ability"),
                    "ga_inferred": ("general_awareness", "Inferred General Awareness"),
                    "math_inferred": ("numerical_ability", "Inferred Numerical Ability"),
                    "math_rescued": ("numerical_ability", "Inferred Numerical Ability"),
                    "non_verbal_figure": ("reasoning", "Inferred Reasoning"),
                    "inferred_gap": ("unknown", "Inferred Gap"),
                    "unknown_gap": ("unknown", "Unknown Gap"),
                }
                section, topic = zone_sections.get(question_type, ("unknown", "unknown"))
            return ClassificationResult(
                topic=topic,
                section=section,
                confidence=0.20,  # Low confidence - we only know the zone
                method="safe-mode-zone-enforced"
            )
        
        # Detect section if not provided
        if not section_hint:
            section_hint = self._detect_section(question_text)
            
        # Get candidate topics for this section
        if section_hint and section_hint in self.TOPIC_TAXONOMY:
            candidate_topics = self.TOPIC_TAXONOMY[section_hint]["topics"]
            sections_to_check = [section_hint]
        else:
            candidate_topics = []
            sections_to_check = list(self.TOPIC_TAXONOMY.keys())
            for section_data in self.TOPIC_TAXONOMY.values():
                candidate_topics.extend(section_data["topics"])
                
        # First: transformer (if enabled)
        if self.use_transformers and self._classifier:
            result = self._classify_with_transformer(question_text, candidate_topics)
            if result.confidence > 0.5:
                result.section = self._get_section_for_topic(result.topic)
                result.method = "zero-shot-transformer"
                if self.total_questions >= 80 and question_number:
                    result = self._apply_zone_enforcement(result, question_number)
                return result
        
        # Second: keywords
        result = self._classify_with_keywords(question_text, sections_to_check)
        result.method = "keyword-matching"
        if self.total_questions >= 80 and question_number:
            result = self._apply_zone_enforcement(result, question_number)

        # Single-section override for short (~30Q) papers
        if target_section and self.total_questions < 80:
            if result.section != target_section:
                result.section = target_section
                result.method = f"{result.method}+single-section"
            section_topics = self.TOPIC_TAXONOMY.get(target_section, {}).get("topics", {})
            if result.topic not in section_topics:
                inferred_topics = {
                    "verbal_ability": "Inferred Verbal Ability",
                    "general_awareness": "Inferred General Awareness",
                    "reasoning": "Inferred Reasoning",
                    "numerical_ability": "Inferred Numerical Ability",
                }
                result.topic = inferred_topics.get(target_section, result.topic)
                result.subtopic = None
                result.confidence = min(result.confidence, 0.35)
        
        # For batch processing: SKIP Gemini entirely to avoid rate limits
        # Gemini is disabled by default for extraction - enable manually for single questions
        # The keyword + zone enforcement is sufficient for 99% of questions
        
        # Groq API fallback for low-confidence classifications
        if self.use_groq and (result.confidence < 0.5 or result.topic == "unknown"):
            groq_result = self._classify_with_groq(question_text, question_number)
            if groq_result and groq_result.confidence > result.confidence:
                # Apply zone enforcement to Groq result too
                if self.total_questions >= 80 and question_number:
                    groq_result = self._apply_zone_enforcement(groq_result, question_number)
                return groq_result
        
        # LLM fallback (ollama) still optional
        should_use_llm = use_llm_fallback if use_llm_fallback is not None else self.use_ollama_fallback
        if should_use_llm and (result.confidence < 0.3 or result.topic == "unknown"):
            llm_result = self._classify_with_ollama(question_text, section_hint)
            if llm_result and llm_result.confidence > result.confidence:
                llm_result.method = "ollama-llm"
                return llm_result
        
        return result
    
    def _classify_with_gemini(
        self, 
        question_text: str, 
        question_number: Optional[int] = None
    ) -> Optional[ClassificationResult]:
        """
        Classify a single question using Gemini 2.0 Flash API (new SDK).
        Handles rate limiting (15 RPM free tier).
        """
        if not self._gemini_client:
            return None
        
        # Rate limiting: max 15 requests per minute
        current_time = time.time()
        if current_time - self._last_gemini_call < 4:  # ~15 RPM = 1 every 4 seconds
            time.sleep(4 - (current_time - self._last_gemini_call))
        
        # Build context-aware prompt
        zone_hint = ""
        if question_number and self.total_questions >= 80:
            zone_hint = self._get_zone_hint(question_number)
        
        prompt = f"""Classify this AFCAT exam question.
{zone_hint}

QUESTION (Q{question_number or '?'}):
{question_text[:500]}

RULES:
1. If question asks for synonym/antonym/meaning/idiom/phrase → Verbal Ability
2. If question has math calculations/percentages/ratios → Numerical Ability  
3. If question has patterns/sequences/coding/relations → Reasoning
4. If question asks about facts/events/places/people → General Awareness

OUTPUT (JSON only, no markdown):
{{"section": "...", "topic": "...", "confidence": 0.XX}}

Valid sections: Verbal Ability, General Awareness, Numerical Ability, Reasoning
Be specific with topic (e.g., "Synonyms", "Time and Work", "Blood Relations")
"""
        
        try:
            self._last_gemini_call = time.time()
            self._gemini_call_count += 1
            
            # New SDK: use client.models.generate_content()
            response = self._gemini_client.models.generate_content(
                model=self._gemini_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temp for factual classification
                    response_mime_type="application/json"  # Forces valid JSON
                )
            )
            
            # Parse JSON from response
            text = response.text.strip()
            # Remove markdown code blocks if present
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                section = data.get("section", "unknown").lower().replace(" ", "_")
                topic = data.get("topic", "unknown").lower().replace(" ", "_")
                confidence = float(data.get("confidence", 0.7))
                
                # Normalize section names
                section_map = {
                    "verbal_ability": "verbal_ability",
                    "general_awareness": "general_awareness", 
                    "numerical_ability": "numerical_ability",
                    "reasoning": "reasoning",
                    "english": "verbal_ability",
                    "math": "numerical_ability",
                    "maths": "numerical_ability",
                    "gk": "general_awareness",
                    "ga": "general_awareness"
                }
                section = section_map.get(section, section)
                
                return ClassificationResult(
                    section=section,
                    topic=topic,
                    confidence=confidence,
                    method="gemini-api"
                )
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Rate limited - extract retry delay if available
                logger.warning("Gemini rate limited, switching to gemini-2.0-flash-lite and pausing for 60s.")
                self._gemini_model_name = "gemini-2.0-flash-lite"
                # Temporarily pause Gemini calls
                self._last_gemini_call = time.time() + 60  # Wait 60 seconds before retry
            else:
                logger.warning(f"Gemini API error: {e}")
            
        return None
    
    def _classify_with_groq(
        self, 
        question_text: str, 
        question_number: Optional[int] = None
    ) -> Optional[ClassificationResult]:
        """
        Classify a single question using Groq API (fast LLM inference).
        Better at understanding context and nuanced topic detection.
        """
        if not self.use_groq or not self.groq_api_key or not requests:
            return None
        
        # Rate limiting: max 30 requests per minute for free tier
        current_time = time.time()
        if current_time - self._last_groq_call < 2:
            time.sleep(2 - (current_time - self._last_groq_call))
        
        # Build context-aware prompt
        zone_hint = ""
        if question_number and self.total_questions >= 80:
            zone_hint = self._get_zone_hint(question_number)
        
        prompt = f"""Classify this AFCAT (Air Force Common Admission Test) exam question into section and topic.
{zone_hint}

QUESTION (Q{question_number or '?'}):
{question_text[:500]}

CLASSIFICATION RULES:
1. Verbal Ability: synonyms, antonyms, idioms, phrases, comprehension, grammar, vocabulary, sentence completion/improvement
2. General Awareness: history, geography, polity, economics, current affairs, science, sports, awards, defense
3. Reasoning: analogies, series, coding-decoding, blood relations, directions, seating arrangement, syllogisms, figures
4. Numerical Ability: arithmetic, percentages, ratios, profit-loss, time-work, speed-distance, algebra, geometry

Respond with ONLY valid JSON (no markdown):
{{"section": "verbal_ability|general_awareness|reasoning|numerical_ability", "topic": "specific_topic_name", "confidence": 0.XX}}
"""
        
        try:
            self._last_groq_call = time.time()
            self._groq_call_count += 1
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are an expert at classifying exam questions. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Remove markdown code blocks if present
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                section = data.get("section", "unknown").lower().replace(" ", "_")
                topic = data.get("topic", "unknown").lower().replace(" ", "_")
                confidence = float(data.get("confidence", 0.75))
                
                # Normalize section names
                section_map = {
                    "verbal_ability": "verbal_ability",
                    "general_awareness": "general_awareness", 
                    "numerical_ability": "numerical_ability",
                    "reasoning": "reasoning",
                    "english": "verbal_ability",
                    "math": "numerical_ability",
                    "maths": "numerical_ability",
                    "gk": "general_awareness",
                    "ga": "general_awareness"
                }
                section = section_map.get(section, section)
                
                return ClassificationResult(
                    section=section,
                    topic=topic,
                    confidence=confidence,
                    method="groq-api"
                )
                
        except Exception as e:
            logger.warning(f"Groq API error: {e}")
            
        return None
    
    def _get_zone_hint(self, question_number: int) -> str:
        """Get zone context hint for Gemini prompt."""
        for section, (start, end) in SECTION_ZONES.items():
            if start <= question_number <= end:
                section_name = section.replace("_", " ").title()
                return f"CONTEXT: Q{question_number} is in the {section_name} zone (Q{start}-Q{end}). Consider this when classifying."
        return ""
    
    def _get_zone_section(self, question_number: int) -> Optional[str]:
        """Get expected section based on question number zone."""
        for section, (start, end) in SECTION_ZONES.items():
            if start <= question_number <= end:
                return section
        return None
    
    def _apply_zone_enforcement(
        self,
        result: ClassificationResult,
        question_number: int
    ) -> ClassificationResult:
        """
        Enforce section zones with selectable mode:
        - strict: always force section from question number
        - flex: only override when classifier confidence is low
        - off: never enforce zones
        """
        expected_section = self._get_zone_section(question_number)
        if expected_section is None:
            return result

        mode = (self.zone_mode or DEFAULT_ZONE_MODE).lower()
        if mode == "off":
            return result

        if mode == "strict":
            if result.section != expected_section:
                logger.debug(
                    f"Q{question_number}: STRICT zone enforcement {result.section} -> {expected_section} "
                    f"(conf={result.confidence:.2f})"
                )
                result.section = expected_section
                result.method = f"{result.method}+strict-zone"
            return result

        # flex mode
        if result.section != expected_section and result.confidence < ZONE_OVERRIDE_THRESHOLD:
            logger.debug(
                f"Q{question_number}: FLEX zone enforcement {result.section} -> {expected_section} "
                f"(conf={result.confidence:.2f})"
            )
            result.section = expected_section
            result.method = f"{result.method}+flex-zone"
            if result.topic == "unknown":
                result.topic = f"inferred_{expected_section}"

        return result
    
    def _detect_section(self, text: str) -> str:
        """
        Detect which AFCAT section a question belongs to.
        Uses enhanced heuristics based on 2026 syllabus patterns.
        """
        text_lower = text.lower()

        # Early verbal guard for typical English patterns to stop GA bleed
        if re.search(r"synonym|antonym|idiom|one word|one-word|substitution|cloze|fill in the blank|fill in the blanks|error detection|no error|spot the error|improve the sentence|jumbled sentence|para jumble|rearrange the sentence|passage|comprehension", text_lower):
            return "verbal_ability"
        if re.search(r"_ _|___|\[\s*\]", text):  # blanks often signal verbal cloze
            return "verbal_ability"
        
        # ========== NUMERICAL ABILITY INDICATORS ==========
        numerical_score = 0
        
        # Strong: Mathematical operators and expressions
        if re.search(r'\d+\s*[+\-×÷*/=<>%]\s*\d+', text):
            numerical_score += 4
        if re.search(r'\d+\s*:\s*\d+', text):  # Ratio format
            numerical_score += 3
        
        # Strong: Mathematical keywords
        numerical_strong = [
            'calculate', 'find the value', 'solve', 'simplify', 'evaluate',
            'percent', '%', 'profit', 'loss', 'interest', 'principal',
            'speed', 'km/hr', 'distance', 'time taken', 'average',
            'ratio', 'proportion', 'lcm', 'hcf', 'divisible',
            'area', 'volume', 'perimeter', 'circumference',
            'probability', 'permutation', 'combination'
        ]
        if any(kw in text_lower for kw in numerical_strong):
            numerical_score += 3
        
        # Medium: Numerical context
        numerical_medium = [
            'rs.', 'rupees', '₹', 'cost', 'price', 'discount',
            'train', 'pipe', 'tank', 'cistern', 'work', 'days',
            'men', 'efficiency', 'fraction', 'decimal'
        ]
        if any(kw in text_lower for kw in numerical_medium):
            numerical_score += 2
        
        # High digit density suggests numerical
        digit_count = sum(c.isdigit() for c in text)
        if digit_count > 8:
            numerical_score += 2
        elif digit_count > 4:
            numerical_score += 1
        
        # ========== VERBAL ABILITY INDICATORS ==========
        verbal_score = 0
        
        # Strong: English language keywords
        verbal_strong = [
            'synonym', 'antonym', 'meaning of', 'opposite of',
            'idiom', 'phrase means', 'one word', 'substitution',
            'error', 'grammatical', 'no error', 'underlined',
            'rearrange', 'jumbled', 'passage', 'comprehension',
            'cloze', 'fill in the blank', 'blank', 'appropriate word',
            'spelling', 'spelt', 'correctly', 'improve the sentence'
        ]
        if any(kw in text_lower for kw in verbal_strong):
            verbal_score += 4
        
        # Medium: Language context
        verbal_medium = [
            'sentence', 'word', 'grammar', 'vocabulary', 'speech',
            'active', 'passive', 'voice', 'direct', 'indirect',
            'paragraph', 'author', 'read the following'
        ]
        if any(kw in text_lower for kw in verbal_medium):
            verbal_score += 2
        
        # Passage indicator (high weight)
        if re.search(r'read the (following )?passage|based on the passage|according to the passage', text_lower):
            verbal_score += 5
        
        # ========== REASONING INDICATORS ==========
        reasoning_score = 0
        
        # Strong: Reasoning keywords
        reasoning_strong = [
            'coded', 'code', 'decode', 'cipher', 'written as',
            'analogy', 'is to', 'as', 'same way',
            'father', 'mother', 'brother', 'sister', 'relation',
            'north', 'south', 'east', 'west', 'direction', 'facing',
            'conclusion', 'follows', 'syllogism', 'premises',
            'venn', 'diagram', 'circles',
            'mirror', 'water image', 'reflection',
            'embedded', 'hidden figure', 'pattern',
            'fold', 'unfold', 'paper', 'punch',
            'cube', 'dice', 'faces', 'opposite'
        ]
        if any(kw in text_lower for kw in reasoning_strong):
            reasoning_score += 4
        
        # Medium: Reasoning context
        reasoning_medium = [
            'series', 'next', 'missing', 'complete the',
            'odd one', 'different', 'does not belong',
            'clock', 'minute', 'hour', 'hands',
            'calendar', 'day', 'week', 'leap year',
            'rank', 'position', 'seating', 'arrangement',
            'sufficient', 'data sufficiency', 'statement alone'
        ]
        if any(kw in text_lower for kw in reasoning_medium):
            reasoning_score += 2
        
        # Figure/diagram reference suggests reasoning
        if re.search(r'figure|diagram|image|pattern|shape', text_lower):
            reasoning_score += 2
        
        # ========== GENERAL AWARENESS INDICATORS ==========
        gk_score = 0
        
        # Strong: GK keywords
        gk_strong = [
            'capital', 'president', 'prime minister', 'governor',
            'constitution', 'article', 'amendment', 'parliament',
            'trophy', 'award', 'prize', 'padma', 'bharat ratna',
            'olympics', 'world cup', 'tournament', 'champion',
            'river', 'mountain', 'country', 'continent',
            'dynasty', 'emperor', 'battle', 'independence',
            'gdp', 'budget', 'rbi', 'economy', 'inflation',
            'launched', 'inaugurated', 'summit', 'agreement',
            'army', 'navy', 'air force', 'missile', 'defence'
        ]
        if any(kw in text_lower for kw in gk_strong):
            gk_score += 4
        
        # Medium: GK context  
        gk_medium = [
            'who', 'which', 'where', 'when', 'established', 'founded',
            'located', 'headquarters', 'first', 'largest', 'longest',
            'discovered', 'invented', 'founded', 'born',
            'festival', 'dance', 'music', 'culture', 'heritage'
        ]
        if any(kw in text_lower for kw in gk_medium):
            gk_score += 2
        
        # Year references often indicate GK/current affairs
        if re.search(r'\b(19|20)\d{2}\b', text):
            gk_score += 1
        
        # ========== DETERMINE SECTION ==========
        scores = {
            "numerical_ability": numerical_score,
            "verbal_ability": verbal_score,
            "reasoning": reasoning_score,
            "general_awareness": gk_score
        }
        
        max_section = max(scores, key=scores.get)
        max_score = scores[max_section]
        
        # Log for debugging
        logger.debug(f"Section scores: {scores}")
        
        # If no clear winner (score < 2), try to infer from structure
        if max_score < 2:
            # Check if it looks like a factual question
            if re.search(r'^(who|what|which|where|when|how many|name the)\b', text_lower):
                return "general_awareness"
            # Check if it has mathematical structure
            elif re.search(r'\d', text) and len(text) < 200:
                return "numerical_ability"
            # Default to GK for ambiguous cases
            return "general_awareness"
            
        return max_section
    
    def _classify_with_transformer(
        self,
        text: str,
        candidate_topics: List[str]
    ) -> ClassificationResult:
        """Use zero-shot classification with transformers."""
        # Format topics for natural language
        topic_labels = [self.TOPIC_LABELS.get(t, t.replace('_', ' ')) 
                        for t in candidate_topics]
        
        try:
            result = self._classifier(
                text[:512],  # Limit length
                candidate_labels=topic_labels,
                multi_label=False
            )
            
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            # Map back to topic key
            for key, label in self.TOPIC_LABELS.items():
                if label == top_label:
                    topic = key
                    break
            else:
                topic = top_label.lower().replace(' ', '_')
            
            # Get alternate topics
            alternates = [
                (self._label_to_topic(l), s)
                for l, s in zip(result['labels'][1:4], result['scores'][1:4])
            ]
            
            return ClassificationResult(
                section=self._get_section_for_topic(topic),
                topic=topic,
                confidence=top_score,
                alternate_topics=alternates
            )
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            return ClassificationResult(
                section="unknown",
                topic="unknown",
                confidence=0.0
            )
    
    def _label_to_topic(self, label: str) -> str:
        """Convert human label back to topic key."""
        for key, val in self.TOPIC_LABELS.items():
            if val == label:
                return key
        return label.lower().replace(' ', '_')
    
    def _classify_with_keywords(
        self,
        text: str,
        sections: List[str]
    ) -> ClassificationResult:
        """
        Enhanced keyword-based classification using AFCAT_CLASSIFICATION dictionary.
        Uses weighted scoring and section-aware topic matching.
        """
        text_lower = text.lower()
        best_match = ("unknown", "unknown", "unknown", 0.0)  # (section, topic_code, topic_name, score)
        all_scores = []
        
        # First: Check for non-verbal reasoning patterns
        is_non_verbal = self._detect_non_verbal(text_lower)
        
        # Use new AFCAT_CLASSIFICATION for enhanced matching
        for section in sections:
            if section not in AFCAT_CLASSIFICATION:
                continue
            
            section_data = AFCAT_CLASSIFICATION[section]
            topics = section_data.get("topics", {})
            
            for topic_code, topic_info in topics.items():
                keywords = topic_info.get("keywords", [])
                weight = topic_info.get("weight", 1.0)
                topic_name = topic_info.get("name", topic_code)
                
                # Count keyword matches with exact word boundary bonus
                matches = 0
                matched_keywords = []
                
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower in text_lower:
                        matches += 1
                        matched_keywords.append(kw)
                        
                        # Bonus for exact word matches (not substrings)
                        if re.search(rf'\b{re.escape(kw_lower)}\b', text_lower):
                            matches += 0.5
                
                if matches > 0:
                    # Calculate weighted score
                    base_score = (matches / max(len(keywords), 1)) * 2
                    weighted_score = min(base_score * weight, 1.0)
                    
                    # Boost non-verbal topics if pattern detected
                    if is_non_verbal and topic_code.startswith("RM_NV_"):
                        weighted_score = min(weighted_score * 1.3, 1.0)
                    
                    all_scores.append((section, topic_code, topic_name, weighted_score))
                    
                    if weighted_score > best_match[3]:
                        best_match = (section, topic_code, topic_name, weighted_score)
        
        # Fallback to legacy TOPIC_TAXONOMY if no match found
        if best_match[3] < 0.1:
            legacy_result = self._classify_with_keywords_legacy(text, sections)
            if legacy_result.confidence > best_match[3]:
                return legacy_result
        
        # Get alternate topics
        all_scores.sort(key=lambda x: x[3], reverse=True)
        alternates = [
            (code, score) for _, code, _, score in all_scores[1:4]
        ] if len(all_scores) > 1 else None
        
        return ClassificationResult(
            section=best_match[0],
            topic=best_match[1],  # topic_code (e.g., VA_SYN, NA_PER)
            confidence=best_match[3],
            subtopic=best_match[2],  # topic_name (e.g., "Synonyms", "Percentages")
            alternate_topics=alternates
        )
    
    def _classify_with_keywords_legacy(
        self,
        text: str,
        sections: List[str]
    ) -> ClassificationResult:
        """Legacy keyword-based classification using TOPIC_TAXONOMY."""
        text_lower = text.lower()
        best_match = ("unknown", "unknown", 0.0)
        all_scores = []
        
        for section in sections:
            if section not in self.TOPIC_TAXONOMY:
                continue
                
            keywords_map = self.TOPIC_TAXONOMY[section].get("keywords", {})
            
            for topic, keywords in keywords_map.items():
                # Count keyword matches
                matches = 0
                matched_keywords = []
                
                for kw in keywords:
                    if kw.lower() in text_lower:
                        matches += 1
                        matched_keywords.append(kw)
                        
                        # Bonus for exact word matches (not substrings)
                        if re.search(rf'\b{re.escape(kw.lower())}\b', text_lower):
                            matches += 0.5
                
                if matches > 0:
                    # Normalize score
                    score = min((matches / len(keywords)) * 2, 1.0)
                    all_scores.append((section, topic, score))
                    
                    if score > best_match[2]:
                        best_match = (section, topic, score)
        
        # Get alternate topics
        all_scores.sort(key=lambda x: x[2], reverse=True)
        alternates = [
            (t, s) for _, t, s in all_scores[1:4]
        ] if len(all_scores) > 1 else None
        
        return ClassificationResult(
            section=best_match[0],
            topic=best_match[1],
            confidence=best_match[2],
            alternate_topics=alternates
        )
    
    def _detect_non_verbal(self, text_lower: str) -> bool:
        """
        Detect if question is likely a non-verbal/figure-based reasoning question.
        Returns True if non-verbal patterns are found.
        """
        for pattern in NON_VERBAL_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def get_topic_code(self, section: str, topic_name: str) -> Optional[str]:
        """
        Get the topic code (e.g., VA_SYN) from section and topic name.
        """
        if section not in AFCAT_CLASSIFICATION:
            return None
        
        topics = AFCAT_CLASSIFICATION[section].get("topics", {})
        topic_name_lower = topic_name.lower()
        
        # Direct match by name
        for code, info in topics.items():
            if info.get("name", "").lower() == topic_name_lower:
                return code
        
        # Fuzzy match
        for code, info in topics.items():
            if topic_name_lower in info.get("name", "").lower():
                return code
        
        return None
    
    def get_section_from_code(self, topic_code: str) -> Optional[str]:
        """
        Get section name from topic code prefix.
        VA_ -> verbal_ability, GA_ -> general_awareness, 
        RM_ -> reasoning, NA_ -> numerical_ability
        """
        prefix_map = {
            "VA_": "verbal_ability",
            "GA_": "general_awareness",
            "RM_": "reasoning",
            "NA_": "numerical_ability"
        }
        for prefix, section in prefix_map.items():
            if topic_code.startswith(prefix):
                return section
        return None
    
    def normalize_topic_name(self, topic_code: str) -> str:
        """
        Convert topic code to human-readable name.
        e.g., VA_SYN -> "Synonyms", NA_PER -> "Percentages"
        """
        for section_data in AFCAT_CLASSIFICATION.values():
            topics = section_data.get("topics", {})
            if topic_code in topics:
                return topics[topic_code].get("name", topic_code)
        
        # Fallback to legacy labels
        return self.TOPIC_LABELS.get(topic_code, topic_code.replace("_", " ").title())
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is not None:
            return self._ollama_available
        
        if requests is None:
            self._ollama_available = False
            return False
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            self._ollama_available = response.status_code == 200
        except Exception:
            self._ollama_available = False
        
        return self._ollama_available
    
    def _classify_with_ollama(
        self,
        text: str,
        section_hint: Optional[str] = None
    ) -> Optional[ClassificationResult]:
        """
        Use Ollama LLM for classification when keywords fail.
        Returns None if Ollama is not available or classification fails.
        """
        if not self._check_ollama_available() or requests is None:
            return None
        
        try:
            
            # Build topic options based on section
            if section_hint and section_hint in self.TOPIC_TAXONOMY:
                topics = self.TOPIC_TAXONOMY[section_hint]["topics"]
                sections = [section_hint]
            else:
                topics = []
                sections = list(self.TOPIC_TAXONOMY.keys())
                for section_data in self.TOPIC_TAXONOMY.values():
                    topics.extend(section_data["topics"])
            
            # Format topics for prompt
            topic_list = ", ".join([self.TOPIC_LABELS.get(t, t) for t in topics[:30]])  # Limit to 30
            section_list = ", ".join(sections)
            
            prompt = f"""Classify this AFCAT exam question into the most appropriate topic.

Question: {text[:500]}

Available sections: {section_list}
Available topics: {topic_list}

Respond with ONLY a JSON object in this exact format (no other text):
{{"section": "section_name", "topic": "topic_name", "confidence": 0.8}}

Use the exact topic names from the list above."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent classification
                        "num_predict": 100
                    }
                },
                timeout=10  # Reduced timeout for faster fallback
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Parse JSON from response
                try:
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{[^}]+\}', response_text)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        
                        section = parsed.get("section", "unknown")
                        topic_label = parsed.get("topic", "unknown")
                        confidence = float(parsed.get("confidence", 0.5))
                        
                        # Map label back to key
                        topic = topic_label.lower().replace(' ', '_').replace('&', 'and')
                        for key, label in self.TOPIC_LABELS.items():
                            if label.lower() == topic_label.lower():
                                topic = key
                                break
                        
                        # Validate section
                        if section not in self.TOPIC_TAXONOMY:
                            section = self._get_section_for_topic(topic)
                        
                        logger.debug(f"Ollama classified as: {section}/{topic} ({confidence})")
                        
                        return ClassificationResult(
                            section=section,
                            topic=topic,
                            confidence=min(confidence, 0.85),  # Cap confidence
                            method="ollama-llm"
                        )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.debug(f"Failed to parse Ollama response: {e}")
                    
        except requests.exceptions.Timeout:
            # Disable Ollama for this session on timeout
            logger.info("Ollama timed out, disabling LLM fallback for this session")
            self._ollama_available = False
        except Exception as e:
            logger.debug(f"Ollama classification failed: {e}")
        
        return None
    
    def _get_section_for_topic(self, topic: str) -> str:
        """Map topic to section (handles both new codes and legacy topics)."""
        # Check new AFCAT_CLASSIFICATION first
        section_from_code = self.get_section_from_code(topic)
        if section_from_code:
            return section_from_code
        
        # Fall back to legacy TOPIC_TAXONOMY
        for section, data in self.TOPIC_TAXONOMY.items():
            if topic in data["topics"]:
                return section
        return "general_awareness"
    
    def classify_batch(
        self,
        questions: List[str],
        sections: Optional[List[str]] = None
    ) -> List[ClassificationResult]:
        """Classify multiple questions."""
        results = []
        sections = sections or [None] * len(questions)
        
        for q, s in zip(questions, sections):
            results.append(self.classify(q, s))
            
        return results
    
    def get_topic_label(self, topic: str) -> str:
        """
        Get human-readable label for a topic.
        Handles both new topic codes (VA_SYN, NA_PER) and legacy topics.
        """
        # Check new AFCAT_CLASSIFICATION first
        for section_data in AFCAT_CLASSIFICATION.values():
            topics = section_data.get("topics", {})
            if topic in topics:
                return topics[topic].get("name", topic)
        
        # Fall back to legacy TOPIC_LABELS
        return self.TOPIC_LABELS.get(topic, topic.replace('_', ' ').title())


class QuestionTypeClassifier:
    """
    Classifies question types: calculation, conceptual, factual, application.
    """
    
    TYPE_INDICATORS = {
        "calculation": {
            "patterns": [
                r'\d+\s*[+\-×÷*/]\s*\d+',
                r'find\s+the\s+value',
                r'calculate',
                r'what\s+is\s+the\s+(?:sum|difference|product|ratio)',
                r'how\s+(?:much|many)',
                r'solve',
            ],
            "keywords": [
                "calculate", "find", "compute", "solve", "value of",
                "equal to", "what is", "how much", "how many"
            ]
        },
        "conceptual": {
            "patterns": [
                r'what\s+is\s+(?:the\s+)?(?:meaning|definition)',
                r'explain',
                r'why\s+(?:is|does|do)',
                r'difference\s+between',
            ],
            "keywords": [
                "concept", "theory", "principle", "law", "define",
                "explain", "what is", "why", "difference between"
            ]
        },
        "factual": {
            "patterns": [
                r'who\s+(?:is|was|are|were)',
                r'when\s+(?:was|did|is)',
                r'where\s+(?:is|was)',
                r'which\s+(?:is|was)',
                r'capital\s+of',
                r'headquarters',
            ],
            "keywords": [
                "who", "when", "where", "capital", "founder",
                "inventor", "discovered", "located", "headquarter",
                "established", "born", "died"
            ]
        },
        "application": {
            "patterns": [
                r'if\s+.+\s+then',
                r'in\s+the\s+given\s+(?:figure|diagram)',
                r'according\s+to\s+the\s+passage',
                r'apply',
                r'using\s+the\s+(?:formula|method)',
            ],
            "keywords": [
                "apply", "using", "given that", "if", "scenario",
                "situation", "case", "example", "according to"
            ]
        }
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for q_type, indicators in self.TYPE_INDICATORS.items():
            self.compiled_patterns[q_type] = [
                re.compile(p, re.IGNORECASE) for p in indicators["patterns"]
            ]
            
    def classify(self, question_text: str) -> Tuple[str, float]:
        """
        Classify question type.
        
        Returns:
            Tuple of (question_type, confidence)
        """
        text_lower = question_text.lower()
        scores = {}
        
        for q_type, indicators in self.TYPE_INDICATORS.items():
            score = 0
            
            # Check patterns
            for pattern in self.compiled_patterns[q_type]:
                if pattern.search(question_text):
                    score += 2
                    
            # Check keywords
            for keyword in indicators["keywords"]:
                if keyword.lower() in text_lower:
                    score += 1
                    
            scores[q_type] = score
            
        if not any(scores.values()):
            return ("unknown", 0.0)
            
        best_type = max(scores, key=scores.get)
        max_score = max(scores.values())
        total_score = sum(scores.values()) or 1
        
        confidence = max_score / total_score
        
        return (best_type, min(confidence, 1.0))


class MathFormulaHandler:
    """
    Handles questions with mathematical formulas.
    Detects and normalizes math notation from OCR.
    """
    
    # Common OCR mistakes in math
    OCR_CORRECTIONS = {
        ' x ': ' × ',
        ' X ': ' × ',
        '**': '^',
        'sqrt': '√',
        'pi': 'π',
    }
    
    FORMULA_PATTERNS = [
        r'\d+\s*[+\-×÷\*/\^]\s*\d+',  # Basic operations
        r'[A-Za-z]\s*=\s*[^,]+',       # Variable assignments
        r'\d+\s*:\s*\d+',               # Ratios
        r'\d+%',                         # Percentages
        r'√\d+',                         # Square roots
        r'\d+°',                         # Degrees
    ]
    
    def __init__(self):
        self.compiled_patterns = [
            re.compile(p) for p in self.FORMULA_PATTERNS
        ]
        
    def has_formula(self, text: str) -> bool:
        """Check if text contains mathematical formulas."""
        return any(p.search(text) for p in self.compiled_patterns)
    
    def normalize_math(self, text: str) -> str:
        """Normalize mathematical notation from OCR output."""
        result = text
        
        for wrong, correct in self.OCR_CORRECTIONS.items():
            result = result.replace(wrong, correct)
            
        return result
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from question text."""
        pattern = r'[-+]?\d*\.?\d+(?:/\d+)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for m in matches:
            try:
                if '/' in m:
                    num, den = m.split('/')
                    numbers.append(float(num) / float(den))
                else:
                    numbers.append(float(m))
            except ValueError:
                pass
                
        return numbers
    
    def get_operation_types(self, text: str) -> List[str]:
        """Detect mathematical operations in text."""
        operations = []
        
        if re.search(r'[+]', text):
            operations.append("addition")
        if re.search(r'[-−]', text):
            operations.append("subtraction")
        if re.search(r'[×*]', text):
            operations.append("multiplication")
        if re.search(r'[÷/]', text):
            operations.append("division")
        if re.search(r'[\^²³]|power|square|cube', text, re.I):
            operations.append("exponentiation")
        if re.search(r'%|percent', text, re.I):
            operations.append("percentage")
        if re.search(r'√|sqrt|root', text, re.I):
            operations.append("root")
            
        return operations
