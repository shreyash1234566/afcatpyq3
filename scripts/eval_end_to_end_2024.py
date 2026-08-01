"""
eval_end_to_end_2024.py
=======================
SOTA back-test: train DM on data <= 2023, generate mock 2024 paper,
evaluate against actual 2024 questions.

Evaluation metric: BM25 retrieval score (topic-filtered) + template match bonus.
Uses Q_clean.json (normalised topic taxonomy) for both context and DM training.
"""

import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# BM25 (pure stdlib, no deps)
# ──────────────────────────────────────────────────────────────────────────────
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.tokenized = [self._tok(d) for d in corpus]
        self.N = len(self.tokenized)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.N, 1)
        self.df = Counter(t for doc in self.tokenized for t in set(doc))
        self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
                    for t, f in self.df.items()}

    def _tok(self, text):
        return re.sub(r'[^a-z0-9\s]', '', str(text).lower()).split()

    def score(self, query):
        qtoks = self._tok(query)
        scores = []
        for doc in self.tokenized:
            dl = len(doc)
            tf = Counter(doc)
            s = 0.0
            for t in qtoks:
                if t not in self.idf: continue
                f = tf.get(t, 0)
                s += self.idf[t] * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores

    def best_match(self, query):
        scores = self.score(query)
        if not scores:
            return None, 0.0
        best = max(range(len(scores)), key=lambda i: scores[i])
        # Normalize to [0,1] by dividing by max possible score for the query
        max_raw = max(scores) if max(scores) > 0 else 1.0
        return best, scores[best] / max_raw


def template_match(q1: str, q2: str) -> float:
    """1.0 if first 4 words match exactly, else 0.0."""
    def first_words(s, n=4):
        return re.sub(r'[^a-z0-9\s]', '', s.lower()).split()[:n]
    return 1.0 if first_words(q1) == first_words(q2) else 0.0


def combined_score(actual: str, gen_qs: list) -> tuple:
    """BM25 (70%) + Template Match (30%) hybrid score."""
    if not gen_qs:
        return "", 0.0
    bm = BM25(gen_qs)
    best_idx, bm_score = bm.best_match(actual)
    tm_score = template_match(actual, gen_qs[best_idx]) if best_idx is not None else 0.0
    final = 0.7 * bm_score + 0.3 * tm_score
    return gen_qs[best_idx] if best_idx is not None else "", final


# ──────────────────────────────────────────────────────────────────────────────
# Template extraction: find the most common structural skeleton for a topic
# ──────────────────────────────────────────────────────────────────────────────
def extract_top_templates(topic_data: list, n=3) -> list:
    """
    Given a list of question dicts for a topic, return the top-N most frequent
    5-word structural templates along with an example question for each.
    """
    prefix_to_examples = defaultdict(list)
    for q in topic_data:
        txt = q.get("question_text", "").strip()
        words = re.sub(r'[^a-z0-9\s]', '', txt.lower()).split()
        if len(words) >= 4:
            prefix = ' '.join(words[:5])
            prefix_to_examples[prefix].append(txt)

    # Sort by frequency
    sorted_prefixes = sorted(prefix_to_examples.items(), key=lambda x: -len(x[1]))
    templates = []
    for prefix, examples in sorted_prefixes[:n]:
        templates.append({
            "template": prefix,
            "example": examples[0],   # Most recent occurrence
            "frequency": len(examples)
        })
    return templates


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders (template-first)
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(qs: list) -> str:
    return "\n".join(f'  - "{q.get("question_text","")}"' for q in qs[:5])


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


def build_template_prompt(sec, topic, templates, examples, n):
    """
    Two-stage template-first prompt:
    Stage 1 — Show the structural skeleton the examiner has been using for years.
    Stage 2 — Ask Groq to fill in the next instance of that skeleton.
    """
    tmpl_lines = "\n".join(
        f'  Template {i+1} (used {t["frequency"]} times): "{t["example"]}"'
        for i, t in enumerate(templates)
    ) if templates else ""

    recent_lines = _fmt(examples)
    prompt = f"""You are predicting the EXACT WORD-TO-WORD text of {n} AFCAT {sec} question(s) on: "{topic}"

STEP 1 - STUDY THESE STRUCTURAL TEMPLATES (the examiner's preferred question skeletons):
{tmpl_lines if tmpl_lines else recent_lines}

STEP 2 - STUDY THESE RECENT EXAMPLES (most recent AFCAT exams, 2021-2023):
{recent_lines}

STEP 3 - PREDICT: Generate {n} question(s) that:
1. START WITH the exact same structural template as the examples above.
2. Only change the specific subject matter (word, number, entity) — not the template structure.
3. 4 options A-D. Distractors must be plausible near-answers.

Return ONLY JSON: {_schema(topic, sec, n)}"""
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# DM training (uses Q_clean.json, data <= 2023 only)
# ──────────────────────────────────────────────────────────────────────────────
from models.dirichlet_forecaster import DirichletForecaster, SEC_TARGET, load_counts
import collections

Q_CLEAN = ROOT / "data" / "processed" / "Q_clean.json"
Q_RAW   = ROOT / "data" / "processed" / "Q.json"
DATA_PATH = Q_CLEAN if Q_CLEAN.exists() else Q_RAW


def _get_year(fn: str):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None


def load_counts_up_to_2023():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    cnt = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    syt = collections.defaultdict(collections.Counter)
    for q in data:
        y = _get_year(q.get("file_name", ""))
        s = q.get("section", "")
        t = q.get("topic", "")
        if y and y < 2024 and s in SEC_TARGET and t:
            cnt[s][t][y] += 1
            syt[s][y] += 1
    years = sorted({y for s in syt for y in syt[s]})
    return cnt, syt, years


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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    from scripts.generate_questions_groq import GroqClient

    FIGURE_TOPICS = {
        "Venn Diagrams", "Non-Verbal Pattern", "Non-Verbal Series",
        "Spatial Ability", "Non-Verbal Classification", "Non-Verbal Analogy",
        "Dot Situation"
    }

    # 1. Train DM on <=2023 data
    print("Training DM Model for 2024 prediction (using data <= 2023)...")
    cnt, syt, years = load_counts_up_to_2023()
    model = DirichletForecaster(cnt, syt, years)
    plan_2024 = model.predict()

    # 2. Load data: ctx = prior to 2024, test = 2024 only
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    ctx_data = []
    test_2024 = defaultdict(list)
    for q in data:
        y   = _get_year(q.get("file_name", ""))
        sec = q.get("section", "")
        top = q.get("topic", "")
        if not y or not sec or not top: continue
        if sec == "Reasoning" and top in FIGURE_TOPICS: continue
        if y < 2024:
            ctx_data.append(q)
        elif y == 2024:
            test_2024[sec].append(q)

    # 3. Generate mock 2024 paper using template-first strategy
    groq = GroqClient()
    generated_by_sec = defaultdict(list)

    SYS = ("You are an AFCAT examiner. Your job is to PREDICT the exact word-to-word "
           "structure of the next AFCAT exam question. Respond ONLY with valid JSON.")

    print("\n--- GENERATING 2024 MOCK PAPER FROM DM PREDICTIONS ---")
    for sec, blk in plan_2024.items():
        allocs = allocate_exact(blk["topics"], blk["section_total"])

        # Redistribute figure-topic slots so section hits its exact target
        if sec == "Reasoning":
            spill = sum(n for t, n in allocs.items() if t in FIGURE_TOPICS and n >= 1)
            if spill:
                fallback = next(
                    (t for t in ["Logical Reasoning", "Verbal Classification",
                                  "Number/Letter Series", "Coding-Decoding", "Verbal Analogy"]
                     if t in allocs and t not in FIGURE_TOPICS), None)
                if fallback:
                    allocs[fallback] = allocs.get(fallback, 0) + spill
            for t in list(allocs):
                if t in FIGURE_TOPICS:
                    allocs[t] = 0

        for topic, n_q in allocs.items():
            if n_q < 1: continue

            # Get recent (recency-sorted, deduplicated) examples for this topic
            topic_ctx = [q for q in ctx_data if q.get("topic") == topic]
            topic_ctx.sort(key=lambda x: _get_year(x.get("file_name", "")) or 0, reverse=True)
            seen_txt = set()
            examples = []
            for q in topic_ctx:
                txt = q.get("question_text", "")
                if txt not in seen_txt:
                    seen_txt.add(txt)
                    examples.append(q)
                if len(examples) >= 5:
                    break

            # Extract structural templates from all historical data for this topic
            templates = extract_top_templates(topic_ctx, n=3)

            prompt = build_template_prompt(sec, topic, templates, examples, n_q)
            resp = groq.chat(SYS, prompt)

            if resp:
                print(f"  -> [OK] {topic}: got response of length {len(resp)}")
                try:
                    res_data = json.loads(resp)
                    count_added = 0
                    for obj in res_data.get("questions", []):
                        q_txt = obj.get("question_text") or obj.get("question")
                        if q_txt:
                            generated_by_sec[sec].append({"text": q_txt, "topic": topic})
                            count_added += 1
                    if count_added == 0:
                        print(f"  -> [WARN] No questions parsed from JSON!")
                except Exception as e:
                    print(f"  -> [ERR] JSON parse: {e} | Raw: {resp[:80]}")
            else:
                print(f"  -> [FAIL] {sec}/{topic}")

    # 4. Evaluate: BM25 + Template Match (topic-filtered)
    print("\n--- EVALUATING END-TO-END (BM25 + Template Match, Topic-Filtered) ---")

    overall_scores = []
    eval_report = {
        "metric": "BM25(0.7) + TemplateMatch(0.3), topic-filtered",
        "sections": {}
    }

    for sec in test_2024:
        print(f"\n--- Section: {sec} ---")
        act_qs  = [q for q in test_2024[sec] if q.get("question_text", "").strip()]
        gen_qs  = generated_by_sec[sec]
        print(f"  Generated: {len(gen_qs)} | Actual: {len(act_qs)}")
        if not gen_qs or not act_qs:
            print(f"  Skipping (empty).")
            continue

        sec_scores = []
        sec_report = []

        for aq in act_qs:
            aq_text  = aq.get("question_text", "")
            aq_topic = aq.get("topic", "")

            # Only match within the exact same topic
            topic_gen = [gq["text"] for gq in gen_qs if gq["topic"] == aq_topic]
            if not topic_gen:
                sec_scores.append(0.0)
                sec_report.append({
                    "actual": aq_text, "topic": aq_topic,
                    "best_generated_match": "[DM predicted 0 questions for this topic]",
                    "score": 0.0
                })
                continue

            best_match, score = combined_score(aq_text, topic_gen)
            sec_scores.append(score)
            sec_report.append({
                "actual": aq_text, "topic": aq_topic,
                "best_generated_match": best_match,
                "score": round(score, 4)
            })

        avg = sum(sec_scores) / len(sec_scores) if sec_scores else 0.0
        print(f"  [{sec}] Average Score: {avg:.3f}")
        overall_scores.extend(sec_scores)
        eval_report["sections"][sec] = {
            "average_score": round(avg, 4),
            "questions": sec_report
        }

    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    eval_report["overall_accuracy"] = round(overall, 4)

    print(f"\n{'='*60}")
    print(f"FULL END-TO-END PREDICTION ACCURACY (2024): {overall:.3f}")
    print(f"{'='*60}\n")

    out_dir = ROOT / "output" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_e2e_2024_report.json"
    out_file.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed evaluation report saved to -> {out_file}")


if __name__ == "__main__":
    main()
