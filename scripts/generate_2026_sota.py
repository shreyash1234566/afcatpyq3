"""
scripts/generate_2026_sota.py
==============================
Generate the AFCAT 2026 mock paper using the SOTA pipeline:
  - DM Forecaster trained on ALL data (2011-2024) via Q_clean.json
  - Template-first generation (structural skeleton cloning)
  - Multi-key Groq rotation with 60s reset on rate limit

Output:
  output/generated_questions/practice_2026_sota.json  (raw questions)
  dashboard/data.js                                    (updated dashboard data)

Run:
    python scripts/generate_2026_sota.py
"""

import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dirichlet_forecaster import DirichletForecaster, SEC_TARGET
from scripts.generate_questions_groq import GroqClient

Q_CLEAN = ROOT / "data" / "processed" / "Q_clean.json"
Q_RAW   = ROOT / "data" / "processed" / "Q.json"
DATA_PATH = Q_CLEAN if Q_CLEAN.exists() else Q_RAW

FIGURE_TOPICS = {
    "Venn Diagrams", "Non-Verbal Pattern", "Non-Verbal Series",
    "Spatial Ability", "Non-Verbal Classification", "Non-Verbal Analogy",
    "Dot Situation"
}


def _get_year(fn: str):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None


def extract_top_templates(topic_data: list, n=3) -> list:
    """Find the most-used structural question skeletons for a topic."""
    from collections import Counter
    prefix_counts = Counter()
    prefix_example = {}
    for q in topic_data:
        txt = q.get("question_text", "").strip()
        words = re.sub(r'[^a-z0-9\s]', '', txt.lower()).split()
        if len(words) >= 4:
            prefix = ' '.join(words[:5])
            prefix_counts[prefix] += 1
            if prefix not in prefix_example:
                prefix_example[prefix] = txt
    templates = []
    for prefix, count in prefix_counts.most_common(n):
        templates.append({"template": prefix, "example": prefix_example[prefix], "frequency": count})
    return templates


def _schema(topic, section, n):
    return json.dumps({
        "questions": [
            {
                "question_text": f"<AFCAT {section} - {topic} question {i+1}>",
                "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
                "correct_answer": "A",
                "explanation": "Step-by-step explanation..."
            }
            for i in range(n)
        ]
    })


def _fmt(qs: list) -> str:
    return "\n".join(f'  - "{q.get("question_text","")}"' for q in qs[:5])


def _load_word_bank():
    """Load vocabulary word bank for constrained generation."""
    wb_path = ROOT / "data" / "vocab_word_bank.json"
    if wb_path.exists():
        return json.loads(wb_path.read_text(encoding="utf-8"))
    return None

WORD_BANK = _load_word_bank()


def build_template_prompt(sec, topic, templates, examples, n):
    tmpl_lines = "\n".join(
        f'  Template {i+1} (used {t["frequency"]} times): "{t["example"]}"'
        for i, t in enumerate(templates)
    ) if templates else _fmt(examples)
    recent_lines = _fmt(examples)

    # ── Enhancement A: Vocabulary Word Bank ──
    vocab_constraint = ""
    if topic == "Synonyms/Antonyms" and WORD_BANK:
        all_words = (WORD_BANK["words"].get("repeated_in_afcat", []) +
                     WORD_BANK["words"].get("high_frequency_afcat_level", []) +
                     WORD_BANK["words"].get("recently_asked_2023_2025", []))
        import random
        sample = random.sample(all_words, min(20, len(all_words)))
        vocab_constraint = f"""
CRITICAL CONSTRAINT - VOCABULARY WORD BANK:
You MUST pick the target word from this curated AFCAT word bank (these are real words from past AFCAT exams):
{', '.join(sample)}
Do NOT invent random words. Pick from this list. Prioritize words that sound like they belong in a defence officer exam.
Distractors must be from the SAME semantic field (e.g., if the word means "brave", options should all relate to courage/fear).
"""

    # ── Enhancement B: Grammar Error Constraints ──
    grammar_constraint = ""
    if topic == "Spotting Errors" and WORD_BANK:
        patterns = WORD_BANK.get("grammar_error_patterns", {}).get("patterns", [])
        pattern_str = "\n".join(f"  - {p['name']} ({p['frequency']}): {p['example']}" for p in patterns[:5])
        grammar_constraint = f"""
CRITICAL CONSTRAINT - GRAMMAR ERROR TYPES:
The error in each question MUST be one of these 5 types (used in 95% of real AFCAT papers):
{pattern_str}
Include "No error" as option D in about 20% of questions.
Signal words to use: each, every, neither, as well as, along with (these create Subject-Verb traps).
"""

    # ── Enhancement B: Idiom Bank ──
    idiom_constraint = ""
    if topic == "Idioms & Phrases" and WORD_BANK:
        idioms = WORD_BANK.get("idiom_bank", {}).get("high_frequency", [])
        import random
        sample_idioms = random.sample(idioms, min(8, len(idioms)))
        idiom_constraint = f"""
CRITICAL CONSTRAINT - IDIOM BANK:
Pick idioms from this AFCAT-tested bank:
{chr(10).join(f'  - "{i}"' for i in sample_idioms)}
Distractors must be plausible but WRONG interpretations of the idiom.
"""

    # ── Enhancement E: Distractor-Aware Options ──
    distractor_note = """
DISTRACTOR RULE: All 4 options must be from the SAME semantic field.
Example: If correct answer is "Watchful", wrong options must be "Careless, Indifferent, Reckless" (all about attention) — NOT random unrelated words.
"""

    return f"""You are predicting the EXACT WORD-TO-WORD text of {n} AFCAT {sec} question(s) on: "{topic}"

STEP 1 - STUDY THESE STRUCTURAL TEMPLATES (the examiner's preferred question skeletons):
{tmpl_lines}

STEP 2 - STUDY THESE RECENT EXAMPLES (most recent AFCAT exams, 2022-2025):
{recent_lines}
{vocab_constraint}{grammar_constraint}{idiom_constraint}{distractor_note}
STEP 3 - PREDICT: Generate {n} question(s) that:
1. START WITH the exact same structural template as the examples above.
2. Only change the specific subject matter (word, number, entity) — not the template.
3. 4 options A-D. Distractors must be plausible near-answers from the same semantic field.
4. Include a brief 1-2 sentence explanation for each answer.

Return ONLY JSON: {_schema(topic, sec, n)}"""


VISUAL_TOPICS = {
    "Non-Verbal Pattern", "Venn Diagrams", "Non-Verbal Series",
    "Spatial Ability", "Non-Verbal Classification", "Non-Verbal Analogy",
    "Dot Situation"
}


def build_visual_prompt(topic, examples, n):
    """
    For Non-Verbal / Venn topics the question TEXT is always present in past papers
    even though the answer OPTIONS are images. We use the past paper question text
    verbatim as templates and ask the model to produce the next predicted variant.

    Venn Diagrams:   "Which of the following represents: X, Y, Z?" — fully text.
    Non-Verbal:      "Which answer figure will complete the pattern?" — fully text.
    """
    recent_lines = _fmt(examples)
    venn_note = (
        "\nNOTE: Venn Diagram questions ONLY have a text stem. The options (A/B/C/D) "
        "should be descriptions like 'Diagram where X and Y overlap but not Z'."
    ) if "Venn" in topic else ""
    nvr_note = (
        "\nNOTE: Non-Verbal questions have a text stem only. Write options as "
        "'Figure A: [description]' etc. — the examiner will match these to the actual figures."
    ) if "Non-Verbal" in topic or "Pattern" in topic else ""

    return f"""You are predicting the EXACT WORD-TO-WORD text of {n} AFCAT Reasoning question(s) on: "{topic}"

PAST PAPER EXAMPLES (use these as templates — clone the stem exactly):
{recent_lines}
{venn_note}{nvr_note}

PREDICT {n} question(s) by:
1. Cloning the EXACT question stem template from the examples above.
2. Changing only the specific entities (e.g. Earth/Sun/Moon → Doctor/Nurse/Hospital).
3. Options A-D should be text descriptions matching the question context.

Return ONLY JSON: {_schema(topic, "Reasoning", n)}"""



def allocate_exact(topics_list, section_target):
    allocs = {}
    remainders = []
    for row in topics_list:
        t = row["topic"]
        e = row["expected_count_exact"]
        allocs[t] = int(math.floor(e))
        remainders.append((e - math.floor(e), t))
    shortfall = section_target - sum(allocs.values())
    remainders.sort(reverse=True, key=lambda x: x[0])
    for i in range(shortfall):
        if i < len(remainders):
            allocs[remainders[i][1]] += 1
    return allocs


def main():
    # 1. Load clean data (all years including 2024)
    print(f"Loading data from {DATA_PATH.name}...")
    all_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"  {len(all_data)} questions loaded.")

    # 2. Train DM on ALL available data (2011–2025)
    print("\nTraining DM Forecaster on full history (2011-2025)...")
    model = DirichletForecaster.from_repo()
    plan_2026 = model.predict()

    # ── Hybrid P1-Weighted Plan (70% P1 / 30% DM) ──
    print("\nLoading 2026 Paper 1 (January) for pattern anchoring...")
    p1_path = ROOT / "data" / "papers" / "afcat_2026_questions.json"
    p1_data = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else []
    
    p1_counts = defaultdict(int)
    for q in p1_data:
        p1_counts[(q.get("section"), q.get("topic"))] += 1

    print("\nApplying 70% Paper 1 / 30% DM hybrid weighting...")
    for sec, blk in plan_2026.items():
        for t_row in blk["topics"]:
            topic = t_row["topic"]
            dm_exp = t_row["expected_count_exact"]
            p1_exp = p1_counts.get((sec, topic), 0)
            
            # The hybrid formula
            hybrid_exp = 0.70 * p1_exp + 0.30 * dm_exp
            
            # If P1 had this topic, guarantee at least a small chance.
            # If P1 didn't have it, but DM does, still keep the DM's 30% contribution.
            t_row["expected_count_exact"] = hybrid_exp
            t_row["dm_expected"] = dm_exp
            t_row["p1_actual"] = p1_exp

        # Re-sort by the new hybrid expected count
        blk["topics"].sort(key=lambda x: -x["expected_count_exact"])
    print("\nDM Forecaster 2026 Plan:")
    section_stats = {}
    for sec, blk in plan_2026.items():
        top5 = sorted(blk["topics"], key=lambda x: -x["expected_count_exact"])[:5]
        section_stats[sec] = {
            "total": blk["section_total"],
            "topics": blk["topics"]
        }
        print(f"\n  [{sec}] Total: {blk['section_total']}")
        for t in top5:
            print(f"    {t['topic']}: {t['expected_count_exact']:.1f} (CI: {t['ci90_low']}-{t['ci90_high']})")

    # 3. Generate questions using template-first strategy
    groq = GroqClient()
    SYS = ("You are an AFCAT examiner. Your job is to PREDICT the exact word-to-word "
           "structure of the next AFCAT 2026 exam question. Respond ONLY with valid JSON.")

    generated_questions = []
    generated_by_sec = defaultdict(list)

    print("\n--- GENERATING AFCAT 2026 MOCK PAPER ---")
    for sec, blk in plan_2026.items():
        allocs = allocate_exact(blk["topics"], blk["section_total"])

        print(f"\n[{sec}]")
        for topic, n_q in allocs.items():
            if n_q < 1: continue

            # Get recent examples — prioritize 2026 Paper 1, then fallback to 2024-2025
            p1_ctx = [q for q in p1_data if q.get("topic") == topic]
            topic_ctx = [q for q in all_data if q.get("topic") == topic]
            topic_ctx.sort(key=lambda x: _get_year(x.get("file_name", "")) or 0, reverse=True)
            
            seen_txt = set()
            examples = []
            
            # Add P1 questions first (strongest signal for Paper 2)
            for q in p1_ctx:
                txt = q.get("question_text", "")
                if txt not in seen_txt:
                    seen_txt.add(txt)
                    # Add a tag so the model knows this is from the immediate prior paper
                    q_copy = dict(q)
                    q_copy["question_text"] = "[FROM 2026 PAPER 1] " + txt
                    examples.append(q_copy)
                if len(examples) >= 5:
                    break
                    
            # Fallback to historical data if we need more examples
            for q in topic_ctx:
                if len(examples) >= 5:
                    break
                txt = q.get("question_text", "")
                if txt not in seen_txt:
                    seen_txt.add(txt)
                    examples.append(q)

            # Extract structural templates or use visual prompt
            if topic in VISUAL_TOPICS:
                prompt = build_visual_prompt(topic, examples, n_q)
            else:
                templates = extract_top_templates(topic_ctx, n=3)
                prompt = build_template_prompt(sec, topic, templates, examples, n_q)
            resp = groq.chat(SYS, prompt)

            if resp:
                try:
                    res_data = json.loads(resp)
                    count_added = 0
                    for obj in res_data.get("questions", []):
                        q_txt = obj.get("question_text") or obj.get("question")
                        if not q_txt:
                            continue
                        # Find the DM topic stats
                        topic_row = next((t for t in blk["topics"] if t["topic"] == topic), {})
                        q_record = {
                            "question_text": q_txt,
                            "options": obj.get("options", {}),
                            "correct_answer": obj.get("correct_answer", ""),
                            "explanation": obj.get("explanation", ""),
                            "topic": topic,
                            "section": sec,
                            "difficulty": "medium",
                            "source": "groq-generated-sota",
                            "dm_predicted_count": round(topic_row.get("expected_count_exact", 0), 3),
                            "ci90_low": topic_row.get("ci90_low", 0),
                            "ci90_high": topic_row.get("ci90_high", 0),
                            "dm_share": round(topic_row.get("dm_share", 0), 4),
                        }
                        generated_questions.append(q_record)
                        generated_by_sec[sec].append({"text": q_txt, "topic": topic})
                        count_added += 1
                    print(f"  [OK] {topic}: {count_added} questions")
                except Exception as e:
                    print(f"  [ERR] {topic}: {e}")
            else:
                print(f"  [FAIL] {topic}")

    # 4. Save raw questions
    out_path = ROOT / "output" / "generated_questions" / "practice_2026_august.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(generated_questions, indent=2), encoding="utf-8")
    print(f"\nSUCCESS: Saved {len(generated_questions)} questions to {out_path.name}")

    # 5. Build dashboard data.js
    print("\nUpdating dashboard/data.js ...")
    _update_dashboard(generated_questions, plan_2026, section_stats)
    print("[DONE] Dashboard updated!")


def _update_dashboard(questions, plan_2026, section_stats):
    """Write a fresh dashboard/data.js with the new SOTA questions and model stats."""

    # Convert questions to dashboard format
    ai_mock_questions = []
    for q in questions:
        opts = q.get("options", {})
        # Support both dict {A:..} and list formats
        if isinstance(opts, dict):
            options_list = [opts.get(k, "") for k in ["A", "B", "C", "D"]]
            correct_letter = q.get("correct_answer", "A")
            correct_text = opts.get(correct_letter, options_list[0] if options_list else "")
        else:
            options_list = opts
            correct_text = options_list[0] if options_list else ""

        ai_mock_questions.append({
            "question_text": q["question_text"],
            "options": options_list,
            "correct_answer": correct_text,
            "section": q["section"],
            "topic": q["topic"],
            "predicted_difficulty": q.get("difficulty", "medium"),
            "question_type": "sota-generated",
            "explanation": q.get("explanation", ""),
            "confidence": round(min(0.95, 0.55 + q.get("dm_share", 0.1) * 2), 2),
            "dm_predicted_count": q.get("dm_predicted_count", 0),
            "ci90_low": q.get("ci90_low", 0),
            "ci90_high": q.get("ci90_high", 0),
            "template_cloned": True,
            "model_accuracy_2024_backtest": 0.683
        })

    # Build section topic distributions
    topic_distribution = {}
    for sec, blk in plan_2026.items():
        for t in blk["topics"]:
            count = round(t["expected_count_exact"])
            if count > 0:
                safe_key = t["topic"].lower().replace(" ", "_").replace("/", "_").replace("&", "and").replace(".", "").replace(",", "")
                topic_distribution[safe_key] = count

    # Section distribution
    section_dist = {sec: blk["section_total"] for sec, blk in plan_2026.items()}

    # Rising topics (topics with upward trend — top predicted topics by section)
    rising = []
    stable = []
    declining = []
    for sec, blk in plan_2026.items():
        for t in blk["topics"]:
            topic_key = t["topic"].lower().replace(" ", "_").replace("/", "_").replace("&", "and")
            actual = t.get("p1_actual", 0)
            hist = t.get("dm_expected", 0)
            diff = actual - hist
            
            if diff >= 2.0:
                rising.append(topic_key)
            elif diff <= -2.0:
                declining.append(topic_key)

    # Build the JS object
    dashboard_data = {
        "pyqCount": 2861,
        "modelAccuracy": {
            "backtest_year": 2024,
            "overall_score": 0.683,
            "metric": "BM25 + Template Match (topic-filtered)",
            "section_scores": {
                "Verbal Ability": 0.733,
                "General Awareness": 0.604,
                "Reasoning": 0.629,
                "Numerical Ability": 0.685
            },
            "note": "Trained on 2011-2023 data only. 2024 paper was completely hidden."
        },
        "aiMockQuestions": ai_mock_questions,
        "dmTopicPlan": {
            sec: {
                "section_total": blk["section_total"],
                "topics": [
                    {
                        "topic": t["topic"],
                        "expected_count": round(t["expected_count_exact"], 2),
                        "ci90_low": t["ci90_low"],
                        "ci90_high": t["ci90_high"]
                    }
                    for t in sorted(blk["topics"], key=lambda x: -x["expected_count_exact"])
                    if t["expected_count_exact"] > 0.5
                ]
            }
            for sec, blk in plan_2026.items()
        },
        "predictions": {
            "rising_topics": rising[:3],
            "declining_topics": declining[:3] if declining else [],
            "topic_predictions": topic_distribution,
            "section_distribution": section_dist
        }
    }

    # Dump as pure JSON to be embedded in the script
    json_string = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
    
    # Must start strictly with `const dashboardData = ` to satisfy Vercel's regex
    js_content = f"""const dashboardData = {json_string};

// Ensure globals are populated for both index.html and dashboard.html
const predictionsData = dashboardData.predictions;
if (typeof window !== 'undefined') {{
    window.dashboardData = dashboardData;
    window.predictionsData = predictionsData;
}}
"""

    dash_path = ROOT / "dashboard" / "data.js"
    dash_path.parent.mkdir(parents=True, exist_ok=True)
    dash_path.write_text(js_content, encoding="utf-8")


if __name__ == "__main__":
    main()
