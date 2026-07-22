"""
AFCAT Q.json Topic Normalizer
==============================
Normalizes fragmented topic names in Q.json to canonical AFCAT syllabus topics.
- Reasoning: 80+ variants → 16 canonical topics
- Verbal Ability: ~30 variants → 12 canonical topics
- Adds `has_figure` field for image-dependent questions
- Fixes `topic_code` to match canonical topic
"""

import json
import re
import shutil
from pathlib import Path
from collections import Counter

# =============================================================================
# CANONICAL TOPIC MAPPINGS
# =============================================================================

# ----- REASONING TOPICS (16 canonical) -----
REASONING_TOPIC_MAP = {
    # → Verbal Analogy (RM_VR_ANALOGY)
    "Verbal Analogy": "Verbal Analogy",
    "Analogy (Verbal)": "Verbal Analogy",
    "Analogy": "Verbal Analogy",
    "Verbal Reasoning - Analogy": "Verbal Analogy",
    "Number Analogy": "Verbal Analogy",

    # → Verbal Classification (RM_VR_CLASS)
    "Verbal Classification": "Verbal Classification",
    "Classification (Verbal)": "Verbal Classification",
    "Classification": "Verbal Classification",
    "Verbal Reasoning - Classification": "Verbal Classification",
    "Classification (Jumbled Words)": "Verbal Classification",
    "Classification (Letter)": "Verbal Classification",
    "Classification (Number)": "Verbal Classification",

    # → Coding-Decoding (RM_VR_CODING)
    "Coding-Decoding": "Coding-Decoding",
    "Coding and Decoding": "Coding-Decoding",
    "Coding Decoding": "Coding-Decoding",
    "Verbal Coding": "Coding-Decoding",
    "Verbal Coding-Decoding": "Coding-Decoding",
    "Verbal Reasoning - Coding and Decoding": "Coding-Decoding",

    # → Venn Diagrams (RM_VR_VENN)
    "Venn Diagrams": "Venn Diagrams",
    "Venn Diagram": "Venn Diagrams",
    "Venn Diagram (Verbal)": "Venn Diagrams",
    "Verbal Venn": "Venn Diagrams",
    "Verbal Reasoning - Venn Diagrams": "Venn Diagrams",
    "Non-Verbal Venn Diagrams": "Venn Diagrams",
    "Non-Verbal Venn Diagram": "Venn Diagrams",
    "Venn Diagram Puzzle": "Venn Diagrams",

    # → Syllogism (RM_VR_SYLL)
    "Syllogism": "Syllogism",

    # → Blood Relations (RM_VR_BLOOD)
    "Blood Relations": "Blood Relations",
    "Verbal Reasoning - Blood Relations": "Blood Relations",

    # → Direction Sense (RM_VR_DIR)
    "Direction and Distance": "Direction Sense",
    "Direction & Distance": "Direction Sense",
    "Direction Sense": "Direction Sense",
    "Direction Sense Test": "Direction Sense",
    "Spatial Orientation": "Direction Sense",
    "Non-Verbal Reasoning - Direction and Orientation": "Direction Sense",

    # → Number/Letter Series (RM_VR_SERIES)
    "Verbal Series": "Number/Letter Series",
    "Number Series": "Number/Letter Series",
    "Letter Series": "Number/Letter Series",
    "Letter/Symbol Series": "Number/Letter Series",
    "Series": "Number/Letter Series",
    "Verbal Reasoning - Number Series": "Number/Letter Series",

    # → Logical Reasoning (RM_VR_LOG)
    "Logical Reasoning": "Logical Reasoning",
    "Logical Reasoning (Verbal)": "Logical Reasoning",
    "Verbal Logic": "Logical Reasoning",
    "Verbal Puzzle": "Logical Reasoning",
    "Character Puzzle": "Logical Reasoning",
    "Puzzle": "Logical Reasoning",
    "Seating Arrangement": "Logical Reasoning",
    "Order & Ranking": "Logical Reasoning",
    "Ranking": "Logical Reasoning",
    "Ranking & Ordering": "Logical Reasoning",
    "Mathematical Operations": "Logical Reasoning",
    "Logical Arrangement": "Logical Reasoning",
    "Verbal Reasoning - Logical Reasoning": "Logical Reasoning",
    "Clocks (Mirror Image)": "Logical Reasoning",
    "Non-Verbal Reasoning - Calendar": "Logical Reasoning",

    # → Non-Verbal Series (RM_NV_SERIES)
    "Non-Verbal Series": "Non-Verbal Series",
    "Non-Verbal Figure Series": "Non-Verbal Series",

    # → Non-Verbal Analogy (RM_NV_ANALOGY)
    "Non-Verbal Analogy": "Non-Verbal Analogy",

    # → Non-Verbal Classification (RM_NV_CLASS)
    "Non-Verbal Classification": "Non-Verbal Classification",

    # → Non-Verbal Pattern (RM_NV_PATTERN)
    "Non-Verbal Pattern": "Non-Verbal Pattern",
    "Non-Verbal Pattern Completion": "Non-Verbal Pattern",
    "Non-Verbal Puzzle": "Non-Verbal Pattern",
    "Non-Verbal Figure Formation": "Non-Verbal Pattern",
    "Pattern": "Non-Verbal Pattern",
    "Pattern Completion": "Non-Verbal Pattern",
    "Completion of Pattern": "Non-Verbal Pattern",
    "Pattern Recognition": "Non-Verbal Pattern",
    "Non-Verbal Pattern Recognition": "Non-Verbal Pattern",
    "Non-Verbal Reasoning - Puzzle": "Non-Verbal Pattern",
    "Figural Visuals": "Non-Verbal Pattern",
    "Non-Verbal Reasoning": "Non-Verbal Pattern",

    # → Embedded Figures (RM_NV_EMBEDDED)
    "Non-Verbal Embedded Figures": "Embedded Figures",
    "Non-Verbal Embedded": "Embedded Figures",
    "Non-Verbal Spatial (Embedded)": "Embedded Figures",
    "Embedded Figures": "Embedded Figures",

    # → Dot Situation (RM_NV_DOT)
    "Dot Situation": "Dot Situation",
    "Non-Verbal Dot Situation": "Dot Situation",
    "Non-Verbal Pattern (Dot Situation)": "Dot Situation",

    # → Spatial Ability (RM_NV_SPATIAL)
    "Non-Verbal Orientation": "Spatial Ability",
    "Non-Verbal Spatial": "Spatial Ability",
    "Non-Verbal Spatial Ability": "Spatial Ability",
    "Non-Verbal Spatial Reasoning": "Spatial Ability",
    "Spatial Ability": "Spatial Ability",
    "Spatial Reasoning": "Spatial Ability",
}

# Canonical topic → topic_code mapping
REASONING_CODE_MAP = {
    "Verbal Analogy": "RM_VR_ANALOGY",
    "Verbal Classification": "RM_VR_CLASS",
    "Coding-Decoding": "RM_VR_CODING",
    "Venn Diagrams": "RM_VR_VENN",
    "Syllogism": "RM_VR_SYLL",
    "Blood Relations": "RM_VR_BLOOD",
    "Direction Sense": "RM_VR_DIR",
    "Number/Letter Series": "RM_VR_SERIES",
    "Logical Reasoning": "RM_VR_LOG",
    "Non-Verbal Series": "RM_NV_SERIES",
    "Non-Verbal Analogy": "RM_NV_ANALOGY",
    "Non-Verbal Classification": "RM_NV_CLASS",
    "Non-Verbal Pattern": "RM_NV_PATTERN",
    "Embedded Figures": "RM_NV_EMBEDDED",
    "Dot Situation": "RM_NV_DOT",
    "Spatial Ability": "RM_NV_SPATIAL",
}

# ----- VERBAL ABILITY TOPICS (12 canonical) -----
VERBAL_TOPIC_MAP = {
    # → Reading Comprehension
    "Reading Comprehension": "Reading Comprehension",
    "Comprehension": "Reading Comprehension",

    # → Cloze Test
    "Cloze Test": "Cloze Test",

    # → Synonyms
    "Synonyms": "Synonyms",

    # → Antonyms
    "Antonyms": "Antonyms",

    # → Error Detection
    "Error Detection": "Error Detection",
    "Error Spotting": "Error Detection",
    "Error Detection (Spelling)": "Error Detection",

    # → Sentence Completion
    "Sentence Completion": "Sentence Completion",
    "Sentence Completion (Phrasal Verbs)": "Sentence Completion",
    "Sentence Completion (Conditionals)": "Sentence Completion",
    "Sentence Completion (Vocabulary)": "Sentence Completion",
    "Vocabulary (Fillers)": "Sentence Completion",

    # → Idioms & Phrases
    "Idioms and Phrases": "Idioms & Phrases",
    "Idioms & Phrases": "Idioms & Phrases",
    "Idioms": "Idioms & Phrases",

    # → One Word Substitution
    "One Word Substitution": "One Word Substitution",

    # → Sentence Rearrangement
    "Sentence Rearrangement": "Sentence Rearrangement",

    # → Verbal Analogy (in English section)
    "Verbal Analogy": "Verbal Analogy",
    "Analogy (Verbal)": "Verbal Analogy",
    "Analogy": "Verbal Analogy",

    # → Spelling
    "Spelling": "Spelling",
    "Spelling Test": "Spelling",
    "Grammar (Spellings)": "Spelling",

    # → Grammar
    "Grammar": "Grammar",
    "Tenses": "Grammar",
    "Tense and Grammar": "Grammar",
    "Grammar (Voice)": "Grammar",
    "Prepositions": "Grammar",
}

VERBAL_CODE_MAP = {
    "Reading Comprehension": "VA_COMP",
    "Cloze Test": "VA_CLOZE",
    "Synonyms": "VA_SYN",
    "Antonyms": "VA_ANT",
    "Error Detection": "VA_ERR",
    "Sentence Completion": "VA_SENT",
    "Idioms & Phrases": "VA_IDIOM",
    "One Word Substitution": "VA_OWS",
    "Sentence Rearrangement": "VA_PARA",
    "Verbal Analogy": "VA_ANALOGY",
    "Spelling": "VA_SPELL",
    "Grammar": "VA_GRAM",
}

# =============================================================================
# FIGURE DETECTION KEYWORDS
# =============================================================================
FIGURE_KEYWORDS = [
    r'\bfigure\b', r'\bfig\b', r'\bdiagram\b', r'\bimage\b',
    r'\bpicture\b', r'\bpattern\b', r'\bdot\b', r'\bembedded\b',
    r'\bfolded\b', r'\bfolding\b', r'\bunfolded\b',
    r'\bmirror\b', r'\bwater image\b', r'\breflection\b',
    r'\brotated\b', r'\brotation\b',
    r'\banswer figures\b', r'\bquestion figure\b',
    r'\boption figure\b', r'\btarget figure\b',
    r'\bgiven figure\b', r'\bfigures?\s*\([a-d1-4]\)',
    r'\bpaper\s+(is\s+)?cut\b', r'\bpaper\s+(is\s+)?fold',
    r'\bcube\b', r'\bdice\b', r'\bblock\b',
    r'\bgrid\b', r'\bmatrix\b',
]
FIGURE_PATTERN = re.compile('|'.join(FIGURE_KEYWORDS), re.IGNORECASE)

# Non-verbal topics always have figures
FIGURE_TOPICS = {
    "Non-Verbal Series", "Non-Verbal Analogy", "Non-Verbal Classification",
    "Non-Verbal Pattern", "Embedded Figures", "Dot Situation", "Spatial Ability",
}


def detect_has_figure(question_text, topic):
    """Detect if a question depends on a figure/image."""
    if topic in FIGURE_TOPICS:
        return True
    if FIGURE_PATTERN.search(question_text or ""):
        return True
    # Check if choices reference figures
    return False


def normalize_questions(data):
    """Normalize all questions in the dataset."""
    stats = {
        "total": len(data),
        "reasoning_fixed": 0,
        "verbal_fixed": 0,
        "figure_marked": 0,
        "unmapped_reasoning": [],
        "unmapped_verbal": [],
    }

    for q in data:
        section = q.get("section", "")
        old_topic = q.get("topic", "")
        question_text = q.get("question_text", "")

        if section == "Reasoning":
            if old_topic in REASONING_TOPIC_MAP:
                new_topic = REASONING_TOPIC_MAP[old_topic]
                new_code = REASONING_CODE_MAP[new_topic]
                if old_topic != new_topic or q.get("topic_code") != new_code:
                    q["topic"] = new_topic
                    q["topic_code"] = new_code
                    stats["reasoning_fixed"] += 1
            else:
                stats["unmapped_reasoning"].append(old_topic)

            # Add has_figure field
            has_fig = detect_has_figure(question_text, q.get("topic", ""))
            # Also check choices for figure references
            for choice in q.get("choices", []):
                choice_text = choice.get("text", "")
                if re.search(r'\bfigure\b|\bfig\b|\boption [a-d]\b', choice_text, re.IGNORECASE):
                    has_fig = True
                    break
            q["has_figure"] = has_fig
            if has_fig:
                stats["figure_marked"] += 1

        elif section == "Verbal Ability":
            if old_topic in VERBAL_TOPIC_MAP:
                new_topic = VERBAL_TOPIC_MAP[old_topic]
                new_code = VERBAL_CODE_MAP[new_topic]
                if old_topic != new_topic or q.get("topic_code") != new_code:
                    q["topic"] = new_topic
                    q["topic_code"] = new_code
                    stats["verbal_fixed"] += 1
            else:
                stats["unmapped_verbal"].append(old_topic)

    return data, stats


def print_report(data, stats):
    """Print a summary report of the normalization."""
    print("=" * 70)
    print("  AFCAT Q.json TOPIC NORMALIZATION REPORT")
    print("=" * 70)
    print(f"\n  Total questions: {stats['total']}")
    print(f"  Reasoning topics fixed: {stats['reasoning_fixed']}")
    print(f"  Verbal topics fixed: {stats['verbal_fixed']}")
    print(f"  Questions marked with has_figure: {stats['figure_marked']}")

    if stats["unmapped_reasoning"]:
        print(f"\n  [!] UNMAPPED reasoning topics: {set(stats['unmapped_reasoning'])}")
    if stats["unmapped_verbal"]:
        print(f"\n  [!] UNMAPPED verbal topics: {set(stats['unmapped_verbal'])}")

    # Print final topic distribution
    print("\n" + "-" * 70)
    print("  REASONING TOPICS (after normalization)")
    print("-" * 70)
    reasoning_topics = Counter(
        q["topic"] for q in data if q.get("section") == "Reasoning"
    )
    for topic, count in reasoning_topics.most_common():
        fig_count = sum(
            1 for q in data
            if q.get("section") == "Reasoning"
            and q.get("topic") == topic
            and q.get("has_figure")
        )
        fig_pct = f" (IMG {fig_count}/{count} need images)" if fig_count else ""
        print(f"    {count:4d}  {topic}{fig_pct}")

    print("\n" + "-" * 70)
    print("  VERBAL ABILITY TOPICS (after normalization)")
    print("-" * 70)
    verbal_topics = Counter(
        q["topic"] for q in data if q.get("section") == "Verbal Ability"
    )
    for topic, count in verbal_topics.most_common():
        print(f"    {count:4d}  {topic}")

    print("\n" + "=" * 70)


def main():
    q_json_path = Path(r"e:\afcatpyq3\data\processed\Q.json")

    # Create backup
    backup_path = q_json_path.with_suffix(".json.bak2")
    shutil.copy2(q_json_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")

    # Load data
    with open(q_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize
    data, stats = normalize_questions(data)

    # Print report
    print_report(data, stats)

    # Save
    with open(q_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved normalized Q.json ({len(data)} questions)")


if __name__ == "__main__":
    main()
