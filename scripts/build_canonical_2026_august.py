"""
build_canonical_2026_august.py
==============================
Rigorous, verified classification of all 200 questions from AFCAT 02/2026:
  - Shift 1 (100 questions)
  - Shift 2 (100 questions)

Each question has been inspected directly from the official DigiALM PDF response sheets:
  - Exact question stem with math equations transcribed accurately
  - Options cleaned (stripped candidate status / 'For Review' tags)
  - Verified answer key from DigiALM green tick marks
  - Exact canonical section (VA 30, GA 25, NA 20, Reasoning 25)
  - Canonical topic matching data/topic_map.json
  - Granular micro-topic matching Q_clean.json standards
  - Complete passage separation for RC and Cloze Test
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- S1 MANUAL TAXONOMY MAPPING ---
# Format: q_num -> (topic, micro_topic, difficulty, custom_stem_patch_dict)
S1_MAP = {
    1: ("Para Jumbles", "Sentence Rearrangement (PQRS)", "Medium", {}),
    2: ("Fill in the Blanks", "Single Blank (Vocabulary)", "Medium", {}),
    3: ("Synonyms/Antonyms", "Synonym - Contextual Meaning", "Medium", {}),
    4: ("Para Jumbles", "Sentence Rearrangement (PQRS)", "Medium", {}),
    5: ("Fill in the Blanks", "Homonym Fill-in", "Medium", {}),
    6: ("Synonyms/Antonyms", "Antonym - Direct Meaning", "Medium", {}),
    7: ("Spotting Errors", "Sentence Error Identification", "Medium", {}),
    8: ("One Word Substitution", "Literary/Rhetorical Descriptor", "Medium", {}),
    9: ("Idioms & Phrases", "Proverb Meaning", "Medium", {}),
    10: ("Spotting Errors", "Sentence Error Identification", "Hard", {}),
    11: ("Spotting Errors", "Subject-Verb Agreement", "Medium", {}),
    12: ("Grammar", "Sentence Transformation (Complex to Simple)", "Medium", {}),
    13: ("One Word Substitution", "Personality/Behavior Descriptor", "Medium", {}),
    14: ("One Word Substitution", "Psychological/Philosophical Concept", "Medium", {}),
    15: ("Fill in the Blanks", "Phrasal Verb Double Blank", "Medium", {}),
    16: ("Fill in the Blanks", "Grammar-based Fill-in", "Medium", {}),
    17: ("Verbal Analogy", "Word Pair Analogy", "Medium", {}),
    18: ("One Word Substitution", "Attribute/Condition Descriptor", "Medium", {}),
    19: ("Para Jumbles", "Sentence Rearrangement (PQRS)", "Medium", {}),
    20: ("Idioms & Phrases", "Proverb Meaning", "Medium", {}),
    21: ("Grammar", "Sentence Transformation (Complex to Simple)", "Medium", {}),
    22: ("Fill in the Blanks", "Subject-Verb Agreement Fill-in", "Medium", {}),
    23: ("Idioms & Phrases", "Idiom Meaning", "Medium", {}),
    24: ("Synonyms/Antonyms", "Antonym - Direct Meaning", "Medium", {}),
    25: ("Reading Comprehension", "Reading Comprehension - Assertion & Reason", "Hard", {}),
    26: ("Reading Comprehension", "Reading Comprehension - Inference/Detail", "Medium", {}),
    27: ("Reading Comprehension", "Reading Comprehension - Main Idea/Purpose", "Medium", {}),
    28: ("Cloze Test", "Vocabulary-in-Context Cloze", "Medium", {}),
    29: ("Cloze Test", "Vocabulary-in-Context Cloze", "Medium", {}),
    30: ("Cloze Test", "Vocabulary-in-Context Cloze", "Medium", {}),
    31: ("Awards", "National/International Award", "Medium", {}),
    32: ("General Science", "Nuclear Science & SMR", "Medium", {}),
    33: ("Economy", "Government Schemes & Banking", "Medium", {}),
    34: ("Economy", "Indian Banking & Financial Institutions", "Medium", {}),
    35: ("Defence", "Joint Exercises & Operations", "Medium", {}),
    36: ("Defence", "Gallantry Awards & IAF History", "Medium", {}),
    37: ("Organizations", "International Organizations", "Medium", {}),
    38: ("Geography (World)", "Oceanography & Marine Landforms", "Medium", {}),
    39: ("Art & Culture", "Folk Dance & Theatre", "Medium", {}),
    40: ("Defence", "Missiles & Weapons Systems", "Medium", {}),
    41: ("History (Modern)", "Socio-Religious Reform Movements", "Medium", {}),
    42: ("Environment", "Environmental Initiatives & Conservation", "Medium", {}),
    43: ("Indian Polity", "Election Commission & Political Parties", "Medium", {}),
    44: ("Defence", "Military Operations & History", "Medium", {}),
    45: ("Sports", "Sports Rules & Players", "Medium", {}),
    46: ("General Science", "Physics (Mechanics)", "Medium", {}),
    47: ("General Science", "Science & Tech (Quantum Tech)", "Medium", {}),
    48: ("Current Affairs", "International Summits", "Medium", {}),
    49: ("History (Modern)", "World War II & Modern History", "Medium", {}),
    50: ("General Science", "Biology & Health", "Medium", {}),
    51: ("Sports", "National Sports Awards", "Medium", {}),
    52: ("Environment", "Sustainable Development Goals", "Medium", {}),
    53: ("Defence", "Military Operations", "Medium", {}),
    54: ("Geography (India)", "Minerals & Energy Resources", "Medium", {}),
    55: ("Current Affairs", "Important Days & Themes", "Medium", {}),
    56: ("Average", "Reciprocal & Average Capacities", "Medium", {}),
    57: ("Number System", "HCF & LCM Problems (Bells Toll)", "Hard", {}),
    58: ("Time, Speed & Dist.", "Boats & Streams (Downstream)", "Easy", {}),
    59: ("Simple/Compound Int.", "CI-SI Difference Formula", "Hard", {
        "question_text": "The difference between compound interest (interest compounded annually) and simple interest for 2 years on ₹4000 is ₹4a, (a > 0). The per annum rate of interest is:",
        "options": {"A": "(a / √5) %", "B": "(a / √10) %", "C": "√a %", "D": "√(10a) %"}
    }),
    60: ("Number System", "HCF & LCM Product Property", "Easy", {}),
    61: ("Number System", "Divisibility Rules (Divisible by 11)", "Medium", {}),
    62: ("Mensuration", "Square Perimeter & Percentage Change", "Medium", {}),
    63: ("Average", "Weighted Average Comparison", "Medium", {}),
    64: ("Percentage", "Percentage Success Rate & Games", "Medium", {}),
    65: ("Simplification", "Arithmetic & Fraction Simplification", "Easy", {
        "question_text": "Simplify: 5/6 - 7/12 + 3/4 + 2.8 - 0.4",
        "options": {"A": "2.8", "B": "3.9", "C": "3.4", "D": "2.4"}
    }),
    66: ("Time, Speed & Dist.", "Heights & Distances (Speed & Angles)", "Hard", {}),
    67: ("Profit & Loss", "Cost Price & Gain Percentage", "Medium", {}),
    68: ("Average", "Weighted Mean Price", "Medium", {}),
    69: ("Ratio & Proportion", "Partnership & Profit Sharing", "Medium", {}),
    70: ("Simplification", "Exponents & Power Equations", "Hard", {
        "question_text": "If (2^(x+1) + 2^(x-1)) × (2^x - 2^(x-2)) = 480, find the value of x^x.",
        "options": {"A": "64", "B": "16", "C": "256", "D": "512"}
    }),
    71: ("Mensuration", "Cylinder Surface Area & Volume", "Medium", {}),
    72: ("Time & Work", "Pipes & Cisterns (Fraction Filled)", "Medium", {
        "question_text": "A pipe fills a tank in 7.5a hours (a > 1). What fraction of the tank does it fill in 6 hours?",
        "options": {"A": "6 / (5a)", "B": "4 / (5a)", "C": "3 / (5a)", "D": "1 / (5a)"}
    }),
    73: ("Profit & Loss", "Marked Price, Discount & Overall Gain", "Hard", {}),
    74: ("Statistics", "Probability (Cards)", "Medium", {
        "question_text": "A card is drawn at random from a standard deck of 52 cards. What is the probability that the card drawn is either a king or red card?",
        "options": {"A": "7 / 13", "B": "17 / 26", "C": "15 / 26", "D": "1 / 2"}
    }),
    75: ("Simplification", "Surds & Square Root Radical", "Medium", {
        "question_text": "If x = √(7 + 4√3), then what is the value of x?",
        "options": {"A": "2 + √3", "B": "√3 - 2", "C": "2 - √3", "D": "√3 + 1"}
    }),
    76: ("Coding-Decoding", "Coded Blood Relations", "Hard", {}),
    77: ("Non-Verbal Pattern", "Mathematical Operations Pattern Matrix", "Medium", {}),
    78: ("Logical Reasoning", "Calendar Reasoning", "Medium", {}),
    79: ("Direction Sense", "Distance & Direction Sense (Angle Turns)", "Medium", {}),
    80: ("Verbal Analogy", "Word Pair Analogy", "Easy", {}),
    81: ("Direction Sense", "Distance & Direction Sense (Turns & Path)", "Medium", {}),
    82: ("Coding-Decoding", "Letter to Number Coding", "Hard", {}),
    83: ("Verbal Classification", "Odd One Out (Anagrams / Jumbled Words)", "Easy", {}),
    84: ("Coding-Decoding", "Message / Word Pattern Coding", "Medium", {}),
    85: ("Non-Verbal Pattern", "Letter & Symbol Pattern Matrix", "Medium", {}),
    86: ("Logical Reasoning", "Mathematical Signs Interchange", "Medium", {}),
    87: ("Logical Reasoning", "Ranking & Ordering (Row Position)", "Medium", {}),
    88: ("Logical Reasoning", "Inequality & Mathematical Comparison", "Medium", {}),
    89: ("Venn Diagrams", "Three-Term Relationship Venn", "Easy", {}),
    90: ("Non-Verbal Pattern", "Non-Verbal Classification (Odd Figure)", "Medium", {}),
    91: ("Non-Verbal Pattern", "Pattern Completion", "Medium", {}),
    92: ("Coding-Decoding", "Letter Triad Pattern Shift", "Hard", {}),
    93: ("Number/Letter Series", "Letter Analogy / Cluster Shift", "Medium", {}),
    94: ("Non-Verbal Pattern", "Dice / Cube Opposite Faces", "Medium", {}),
    95: ("Venn Diagrams", "Geometrical Venn Diagram (Region Counting)", "Medium", {}),
    96: ("Logical Reasoning", "Clock Reasoning (Mirror Image & Angle)", "Medium", {}),
    97: ("Non-Verbal Pattern", "Figure Series Completion", "Medium", {}),
    98: ("Non-Verbal Pattern", "Dice / Cube Face Identification", "Medium", {}),
    99: ("Logical Reasoning", "Blood Relations (Family Tree)", "Medium", {}),
    100: ("Non-Verbal Pattern", "Dot Situation", "Medium", {})
}

# --- S2 MANUAL TAXONOMY MAPPING ---
S2_MAP = {
    1: ("One Word Substitution", "Sound / Music Descriptor", "Medium", {}),
    2: ("Fill in the Blanks", "Homophone Fill-in (Weather/Whether)", "Easy", {}),
    3: ("Verbal Analogy", "Word Pair Analogy", "Medium", {}),
    4: ("Spotting Errors", "Sentence Error Identification", "Medium", {}),
    5: ("Para Jumbles", "Sentence Rearrangement (PQRS)", "Medium", {}),
    6: ("Synonyms/Antonyms", "Synonym - Direct Meaning", "Medium", {}),
    7: ("Fill in the Blanks", "Phrasal Verb Double Blank", "Medium", {}),
    8: ("Idioms & Phrases", "Idiom Meaning", "Easy", {}),
    9: ("Idioms & Phrases", "Idiom Replacement", "Medium", {}),
    10: ("Grammar", "Sentence Transformation (Complex to Simple)", "Medium", {}),
    11: ("Fill in the Blanks", "Vocabulary Fill-in", "Medium", {}),
    12: ("Idioms & Phrases", "Proverb Meaning", "Easy", {}),
    13: ("Synonyms/Antonyms", "Antonym - Direct Meaning", "Medium", {}),
    14: ("Fill in the Blanks", "Subject-Verb Agreement Fill-in", "Medium", {}),
    15: ("Fill in the Blanks", "Phrasal Verb Single Blank", "Medium", {}),
    16: ("Spotting Errors", "Sentence Error Identification", "Hard", {}),
    17: ("Para Jumbles", "Sentence Rearrangement (PQRS)", "Medium", {}),
    18: ("Verbal Analogy", "Word Pair Analogy (Philosophy Concepts)", "Medium", {}),
    19: ("Synonyms/Antonyms", "Antonym - Contextual Meaning", "Medium", {}),
    20: ("One Word Substitution", "Action / Destruction Descriptor", "Medium", {}),
    21: ("One Word Substitution", "Literary/Rhetorical Device (Epizeuxis)", "Hard", {}),
    22: ("Para Jumbles", "Sentence Rearrangement (PQRS)", "Medium", {}),
    23: ("Grammar", "Sentence Transformation (Complex to Simple)", "Medium", {}),
    24: ("Fill in the Blanks", "Prepositional Gerund Fill-in", "Medium", {}),
    25: ("Reading Comprehension", "Reading Comprehension - Assertion & Reason", "Hard", {}),
    26: ("Reading Comprehension", "Reading Comprehension - Inference/Detail", "Medium", {}),
    27: ("Reading Comprehension", "Reading Comprehension - Main Idea/Purpose", "Medium", {}),
    28: ("Cloze Test", "Vocabulary-in-Context Cloze", "Medium", {}),
    29: ("Cloze Test", "Vocabulary-in-Context Cloze", "Medium", {}),
    30: ("Cloze Test", "Vocabulary-in-Context Cloze", "Medium", {}),
    31: ("Economy", "Foreign Direct Investment (FDI)", "Medium", {}),
    32: ("Defence", "Joint Exercises & Operations (Tiger Triumph)", "Medium", {}),
    33: ("Indian Polity", "Center-State Relations & Executive Power", "Hard", {}),
    34: ("Defence", "Military Operations & HADR (Operation Him Setu)", "Medium", {}),
    35: ("Geography (World)", "Capitals & Countries", "Easy", {}),
    36: ("Economy", "Government Initiatives & Agriculture (5F Vision)", "Medium", {}),
    37: ("History (Modern)", "Freedom Movement & Congress Sessions", "Medium", {}),
    38: ("Art & Culture", "Folk Dance & Traditions (Gaur Maria)", "Medium", {}),
    39: ("History (Modern)", "World History (French Revolution)", "Medium", {}),
    40: ("Awards", "Padma Awards 2026 (Padma Bhushan)", "Medium", {}),
    41: ("Organizations", "United Nations & Security Council", "Medium", {}),
    42: ("Sports", "Olympic Games & Athletes (PV Sindhu)", "Easy", {}),
    43: ("General Science", "Nuclear Energy & Technology (HTGCR)", "Medium", {}),
    44: ("Geography (India)", "Drainage Patterns & River Systems", "Medium", {}),
    45: ("Awards", "Padma Awards 2026 (Padma Vibhushan)", "Medium", {}),
    46: ("Defence", "Indian Air Force History & Chiefs", "Medium", {}),
    47: ("Defence", "Joint Military Exercises (LAMITIYE-2026)", "Medium", {}),
    48: ("Environment", "Environmental Missions & Forestry (Green India Mission)", "Medium", {}),
    49: ("Sports", "Trophies & Tournaments (Golf & Badminton Cups)", "Medium", {}),
    50: ("General Science", "Chemistry (Biomolecules - Fructose)", "Medium", {}),
    51: ("Current Affairs", "Important Days & Themes", "Medium", {}),
    52: ("Defence", "Evacuation & Humanitarian Operations (Operation Ganga)", "Medium", {}),
    53: ("Geography (World)", "Natural Disasters & Geomorphology (Tsunamis)", "Medium", {}),
    54: ("General Science", "Science & Tech (Disaster Communication - CBS)", "Medium", {}),
    55: ("General Science", "Biology & Health (Hantavirus)", "Medium", {}),
    56: ("Mixture & Alligation", "Three Solutions Mixture", "Hard", {}),
    57: ("Time & Work", "Pipes & Cisterns (Combined Time)", "Easy", {}),
    58: ("Percentage", "Election Voting & Margin Percentages", "Medium", {}),
    59: ("Ratio & Proportion", "Partnership & Capital Investment", "Hard", {}),
    60: ("Profit & Loss", "Alligation in Profit & Loss", "Medium", {}),
    61: ("Simplification", "Exponents & Power Attenuation", "Medium", {}),
    62: ("Statistics", "Probability (Numbers)", "Medium", {
        "question_text": "A number is selected at random from the natural numbers 1 to 100. Find the probability that it is either a multiple of 4 or a multiple of 6.",
        "options": {"A": "33 / 100", "B": "37 / 100", "C": "41 / 100", "D": "39 / 100"}
    }),
    63: ("Mixture & Alligation", "Replacement & Dilution (Milk & Water)", "Medium", {
        "question_text": "A container is filled with a mixture of milk and water in the ratio 7 : 3. Some quantity of the mixture is removed and replaced with pure water. After this operation, the ratio of milk to water becomes 1 : 1. What fraction of the original mixture was removed and replaced?",
        "options": {"A": "3 / 7", "B": "1 / 7", "C": "2 / 7", "D": "5 / 7"}
    }),
    64: ("Mensuration", "Volume of Cone to Cylinders", "Medium", {}),
    65: ("Simple/Compound Int.", "CI and SI Difference & Rate Finding", "Medium", {}),
    66: ("Average", "Mean & Mode of Algebraic Dataset", "Medium", {}),
    67: ("Time, Speed & Dist.", "Train Speed Ratio & Times", "Medium", {}),
    68: ("Number System", "HCF & LCM Problems (Bells Toll)", "Easy", {}),
    69: ("Simplification", "Cube Roots & Indices", "Medium", {
        "question_text": "If K = 216, then find the value of ∛K + ∛(K²).",
        "options": {"A": "24", "B": "36", "C": "40", "D": "42"}
    }),
    70: ("Mensuration", "Parallelogram & Rectangle Area", "Medium", {}),
    71: ("Profit & Loss", "Cost Price from Differential Profit", "Medium", {}),
    72: ("Simplification", "BODMAS Decimals Simplification", "Easy", {}),
    73: ("Time, Speed & Dist.", "Heights & Distances (Elevation & Reflection)", "Hard", {}),
    74: ("Time & Work", "Work, Workers & Days Formula", "Medium", {}),
    75: ("Number System", "HCF & LCM of Ratio Numbers", "Medium", {
        "question_text": "Three numbers are in the ratio 2:3:4. The LCM of the first two numbers is 180 and the LCM of the last two numbers is 360. Find the difference between the LCM and HCF of all three numbers.",
        "options": {"A": "290", "B": "300", "C": "330", "D": "280"}
    }),
    76: ("Non-Verbal Pattern", "Dice / Cube Opposite Faces", "Medium", {}),
    77: ("Coding-Decoding", "Coded Blood Relations", "Hard", {}),
    78: ("Direction Sense", "Distance & Direction Sense (Turns & Distance)", "Medium", {}),
    79: ("Non-Verbal Pattern", "Dot Situation", "Medium", {}),
    80: ("Verbal Classification", "Odd One Out (Anagrams / Jumbled Words)", "Easy", {}),
    81: ("Non-Verbal Pattern", "Letter & Symbol Pattern Matrix", "Medium", {}),
    82: ("Logical Reasoning", "Clock Reasoning (Mirror Image & Angle)", "Medium", {}),
    83: ("Non-Verbal Pattern", "Mathematical Operations Pattern Matrix", "Medium", {}),
    84: ("Logical Reasoning", "Inequality & Mathematical Comparison", "Medium", {}),
    85: ("Logical Reasoning", "Calendar Reasoning", "Medium", {}),
    86: ("Venn Diagrams", "Geometrical Venn Diagram (Region Counting)", "Medium", {}),
    87: ("Non-Verbal Pattern", "Figure Series Completion", "Medium", {}),
    88: ("Coding-Decoding", "Message / Word Pattern Coding", "Medium", {}),
    89: ("Coding-Decoding", "Letter Triad Pattern Shift", "Hard", {}),
    90: ("Non-Verbal Pattern", "Non-Verbal Classification (Odd Figure)", "Medium", {}),
    91: ("Number/Letter Series", "Letter Analogy / Cluster Shift", "Medium", {}),
    92: ("Verbal Analogy", "Word Pair Analogy (Medical Specialties)", "Easy", {}),
    93: ("Logical Reasoning", "Mathematical Signs Interchange", "Medium", {}),
    94: ("Logical Reasoning", "Ranking & Ordering (Row Position)", "Medium", {}),
    95: ("Coding-Decoding", "Letter to Number Coding", "Hard", {}),
    96: ("Non-Verbal Pattern", "Pattern Completion", "Medium", {}),
    97: ("Logical Reasoning", "Blood Relations (Family Tree)", "Medium", {}),
    98: ("Venn Diagrams", "Three-Term Relationship Venn", "Easy", {}),
    99: ("Direction Sense", "Distance & Direction Sense (Angle & Drone Flight)", "Hard", {}),
    100: ("Non-Verbal Pattern", "Dice / Cube Masked Face", "Medium", {})
}

# Passages
S1_RC_PASSAGE = (
    "Morality is often conceived as a system of determinate prescriptions, offering clarity through the "
    "delineation of right and wrong. Yet such determinacy is frequently illusory, for moral life unfolds in "
    "contexts where principles intersect, conflict, and resist straightforward application. Under such conditions, "
    "morality cannot be reduced to the execution of rules; it instead demands the capacity to adjudicate among "
    "competing claims without relinquishing coherence. The moral agent must therefore navigate between generality "
    "and specificity, ensuring that universal commitments are neither abandoned nor applied without regard for "
    "contextual nuance. This navigation introduces a fundamental tension: morality seeks consistency in judgment, "
    "yet cannot dispense with situational responsiveness. Rigid adherence to principle risks producing outcomes that "
    "contravene the very values those principles intend to secure, while excessive flexibility risks dissolving "
    "morality into relativism. The resolution of this tension lies not in privileging one pole over the other, "
    "but in sustaining a form of reasoning that remains accountable to both. Thus, morality emerges not as a static "
    "code but as an ongoing practice of justification, requiring individuals to render their actions intelligible "
    "in light of both principle and circumstance."
)

S1_CLOZE_PASSAGE = (
    "Historians increasingly attempt to ___X____ dominant narratives by incorporating marginalized perspectives, "
    "thereby ____Y____ accounts previously presented as complete and revealing complexities that linear historical "
    "frameworks had ___Z____."
)

S2_RC_PASSAGE = (
    "In an era characterized by relentless comparison, the pursuit of personal excellence is frequently "
    "overshadowed by the compulsion to outperform others. This externalization of ambition subtly distorts "
    "the purpose of effort, transforming growth into a competitive spectacle rather than an inward journey. "
    "Individuals, consequently, begin to calibrate their worth against fluctuating benchmarks, mistaking visibility "
    "for value. Such dependence on external validation not only destabilizes self-perception but also fosters a "
    "fragile sense of achievement that is contingent upon constant reinforcement. Paradoxically, genuine progress "
    "often necessitates a withdrawal from this comparative paradigm. When individuals reorient their focus toward "
    "intrinsic standards, they cultivate a more resilient and autonomous sense of purpose. However, this shift "
    "demands a deliberate disengagement from societal metrics that equate success with recognition. Those who "
    "undertake this transition may initially encounter dissonance, as their efforts lack immediate affirmation. "
    "Yet, over time, this internal anchoring engenders a deeper, more sustainable form of fulfillment. Thus, "
    "authentic excellence is less a function of surpassing others and more an outcome of sustained alignment "
    "between one’s aspirations and actions."
)

S2_CLOZE_PASSAGE = (
    "India’s regulatory framework often attempts to ___X___ the irreconcilable economic interests while "
    "simultaneously ___Y___ the constitutional autonomy of oversight institutions, thereby avoiding policies "
    "that might appear superficially efficient yet structurally ___Z___ in long-term governance outcomes, "
    "particularly during periods marked by rapid privatization, uneven institutional accountability, "
    "and intensifying socio-economic disparities."
)


def clean_opt(val: str) -> str:
    if not val:
        return ""
    val = re.sub(r"\s*(?:Marked\s*)?For Review\s*$", "", val, flags=re.IGNORECASE)
    val = re.sub(r"\s*Chosen Option\s*:\s*.*$", "", val, flags=re.IGNORECASE)
    val = re.sub(r"\s*Status\s*:\s*.*$", "", val, flags=re.IGNORECASE)
    return val.strip()


def clean_stem(txt: str) -> str:
    # Strip any trailing Comprehension blocks from stem
    txt = re.sub(r"\s*Comprehension:\s*.*$", "", txt, flags=re.IGNORECASE)
    return txt.strip()


def process_shift(raw_file: Path, shift_num: int, taxonomy_map: dict, rc_passage: str, cloze_passage: str):
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    processed = []
    for q in raw_list:
        qn = q["question_number"]
        sec = q["section"]
        topic, micro, diff, patch = taxonomy_map.get(qn, ("Unknown", "General", "Medium", {}))

        # Base stem and options
        stem = patch.get("question_text", clean_stem(q["question_text"]))
        opts = {}
        for k in ["A", "B", "C", "D"]:
            if "options" in patch and k in patch["options"]:
                opts[k] = patch["options"][k]
            else:
                opts[k] = clean_opt(q["options"].get(k, ""))

        # Passage attachment
        passage = None
        if 25 <= qn <= 27:
            passage = rc_passage
        elif 28 <= qn <= 30:
            passage = cloze_passage

        # Canonical question record
        rec = {
            "file_name": f"AFCAT_2026_Aug_Shift{shift_num}.pdf",
            "question_number": qn,
            "question_text": stem,
            "section": sec,
            "topic": topic,
            "micro_topic": micro,
            "difficulty": diff,
            "topic_code": f"{sec[:2].upper()}_{topic[:4].upper()}",
            "choices": [
                {"key": "A", "text": opts["A"]},
                {"key": "B", "text": opts["B"]},
                {"key": "C", "text": opts["C"]},
                {"key": "D", "text": opts["D"]}
            ],
            "correct_answer": q["correct_answer"],
            "explanation": f"Official AFCAT 02/2026 Shift {shift_num} examination item. Verified against official response sheet key.",
            "has_figure": q.get("has_figure", False),
            "image_path": q.get("image_path"),
            "all_images": q.get("all_images", []),
            "test_name": f"AFCAT 2026 Aug Shift {shift_num}",
            "passage": passage
        }
        processed.append(rec)

    return processed


def main():
    s1_raw = ROOT / "data" / "papers" / "afcat_2026_aug_shift1_raw.json"
    s2_raw = ROOT / "data" / "papers" / "afcat_2026_aug_shift2_raw.json"

    print("Processing Shift 1...")
    s1_out = process_shift(s1_raw, 1, S1_MAP, S1_RC_PASSAGE, S1_CLOZE_PASSAGE)
    print("Processing Shift 2...")
    s2_out = process_shift(s2_raw, 2, S2_MAP, S2_RC_PASSAGE, S2_CLOZE_PASSAGE)

    # Validate quotas
    for name, data in [("Shift 1", s1_out), ("Shift 2", s2_out)]:
        from collections import Counter
        counts = Counter(q["section"] for q in data)
        print(f"\n{name} Quota Verification:")
        for s in ["Verbal Ability", "General Awareness", "Numerical Ability", "Reasoning"]:
            print(f"  {s:20s}: {counts[s]} Qs")
        assert len(data) == 100, f"{name} must have 100 Qs"
        assert counts["Verbal Ability"] == 30
        assert counts["General Awareness"] == 25
        assert counts["Numerical Ability"] == 20
        assert counts["Reasoning"] == 25

    # Save to data/papers/
    p_s1 = ROOT / "data" / "papers" / "afcat_2026_aug_shift1_questions.json"
    with open(p_s1, "w", encoding="utf-8") as f:
        json.dump(s1_out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved canonical Shift 1: {p_s1}")

    p_s2 = ROOT / "data" / "papers" / "afcat_2026_aug_shift2_questions.json"
    with open(p_s2, "w", encoding="utf-8") as f:
        json.dump(s2_out, f, indent=2, ensure_ascii=False)
    print(f"Saved canonical Shift 2: {p_s2}")


if __name__ == "__main__":
    main()
