"""
evaluate_generation.py
======================
Walk-forward quality evaluation of Groq-generated AFCAT questions.

Protocol (zero leakage):
  For each test year T in [2022, 2023, 2024]:
    - RAG context = only PYQs from years < T
    - Generate N questions per predicted-high-priority topic
    - Compare generated text against actual exam questions from year T
    - Report: semantic similarity, topic coverage, diversity

Metrics:
  - Semantic Similarity  : SBERT cosine(generated, actual_T)
  - Topic Coverage       : % of actual topics that were generated for
  - Intra-set Diversity  : avg pairwise dissimilarity of generated Qs
  - LLM-as-Judge         : Groq scores each Q 1-5 on quality
"""

import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_questions_groq import (
    AFCATQuestionGenerator, SAFE_REASONING_TOPICS,
    GroqClient, GROQ_API_KEY, SYSTEM_PROMPT,
)

Q_JSON_PATH = ROOT / "data" / "processed" / "Q.json"
EVAL_OUTPUT  = ROOT / "output" / "evaluation"
EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_SECTIONS = ["Verbal Ability", "Numerical Ability", "Reasoning", "General Awareness"]
TEST_YEARS      = [2022, 2023, 2024]
QS_PER_TOPIC    = 2   # keep API usage low during eval


# ──────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_year(fn: str):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None


def load_by_year(path=Q_JSON_PATH) -> Dict[int, List[Dict]]:
    data = json.load(open(path, encoding="utf-8"))
    by_year = defaultdict(list)
    for q in data:
        y = _get_year(q.get("file_name", ""))
        if y:
            by_year[y].append(q)
    return by_year


def top_topics(questions: List[Dict], section: str, n: int = 5) -> List[str]:
    """Top-N topics for a section based on question count."""
    counts = Counter(q.get("topic") for q in questions if q.get("section") == section)
    return [t for t, _ in counts.most_common(n) if t]


# ──────────────────────────────────────────────────────────────────────────────
# SBERT SEMANTIC SCORER
# ──────────────────────────────────────────────────────────────────────────────

class SemanticScorer:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-mpnet-base-v2")
            print("[Eval] SBERT model loaded")
        except ImportError:
            self.model = None
            print("[Eval] WARNING: sentence_transformers not found -- semantic scores will be 0.0")

    def score(self, generated: List[str], actual: List[str]) -> Dict:
        """Compute max cosine similarity between each generated Q and any actual Q."""
        if not self.model or not generated or not actual:
            return {"mean_max_sim": 0.0, "median_max_sim": 0.0, "top10_mean": 0.0}

        g_embs = self.model.encode(generated, normalize_embeddings=True)
        a_embs = self.model.encode(actual,    normalize_embeddings=True)
        sim_matrix = g_embs @ a_embs.T   # (G, A)
        max_sims = sim_matrix.max(axis=1)  # for each generated Q: best match in actual

        return {
            "mean_max_sim":   float(max_sims.mean()),
            "median_max_sim": float(np.median(max_sims)),
            "top10_mean":     float(np.sort(max_sims)[-10:].mean()) if len(max_sims) >= 10 else float(max_sims.mean()),
        }

    def diversity(self, texts: List[str]) -> float:
        """Intra-set diversity: 1 - mean pairwise cosine similarity."""
        if not self.model or len(texts) < 2:
            return 0.0
        embs = self.model.encode(texts, normalize_embeddings=True)
        sim  = embs @ embs.T
        n    = len(texts)
        off_diag = (sim.sum() - np.trace(sim)) / (n * (n - 1))
        return float(1.0 - off_diag)


# ──────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = "You are a strict competitive exam quality auditor. Respond ONLY with valid JSON."

def judge_prompt(q: Dict) -> str:
    return f"""Rate this AFCAT MCQ 1-5 on:
- clarity (is the question unambiguous?)
- difficulty (appropriate for a competitive exam?)
- distractor_quality (are wrong options plausible?)
- correctness (is the marked answer actually correct?)

Question:
{json.dumps(q, indent=2)}

Return ONLY:
{{"clarity": 4, "difficulty": 4, "distractor_quality": 3, "correctness": 5, "overall": 4, "comment": "..."}}"""


def llm_judge_batch(questions: List[Dict], groq: GroqClient, sample=5) -> Dict:
    """Score a random sample of questions via Groq."""
    import random
    sample_qs = random.sample(questions, min(sample, len(questions)))
    scores = []
    for q in sample_qs:
        raw = groq.chat(JUDGE_SYSTEM, judge_prompt(q), temperature=0.0, max_tokens=256)
        if not raw:
            continue
        try:
            r = json.loads(raw)
            scores.append(r)
        except Exception:
            pass

    if not scores:
        return {}

    keys = ["clarity", "difficulty", "distractor_quality", "correctness", "overall"]
    avg = {k: float(np.mean([s.get(k, 0) for s in scores])) for k in keys}
    avg["n_judged"] = len(scores)
    return avg


# ──────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(test_years=TEST_YEARS, questions_per_topic=QS_PER_TOPIC):
    by_year = load_by_year()
    scorer  = SemanticScorer()
    groq    = GroqClient()
    all_results = []

    print(f"\n{'='*60}")
    print("AFCAT QUESTION GENERATION — Walk-Forward Evaluation")
    print(f"Test years: {test_years} | Q per topic: {questions_per_topic}")
    print(f"{'='*60}\n")

    for test_year in test_years:
        print(f"\n{'─'*50}")
        print(f"  TEST YEAR: {test_year}  (training on years < {test_year})")
        print(f"{'─'*50}")

        # Instantiate generator using ONLY data before test_year
        gen = AFCATQuestionGenerator(cutoff_year=test_year, use_critique=False)

        # Actual questions from test_year (held-out)
        actual_qs = by_year.get(test_year, [])
        print(f"  Actual exam Qs in {test_year}: {len(actual_qs)}")

        year_result = {"test_year": test_year, "sections": {}}

        for section in TARGET_SECTIONS:
            actual_sec = [q for q in actual_qs if q.get("section") == section and q.get("question_text")]
            if not actual_sec:
                continue

            actual_topics = set(q.get("topic") for q in actual_sec)
            top_t = top_topics([q for y in by_year for q in by_year[y]
                                 if _get_year(by_year[y][0].get("file_name","")) < test_year
                                 if q.get("section") == section], section, n=6)

            # Filter reasoning to text-only
            if section == "Reasoning":
                top_t = [t for t in top_t if t in SAFE_REASONING_TOPICS]

            if not top_t:
                continue

            print(f"\n  [{section}] Generating for {len(top_t)} topics: {top_t}")

            generated_qs = []
            for topic in top_t:
                qs = gen.generate_topic(topic, section, count=questions_per_topic, difficulty="medium")
                generated_qs.extend(qs)

            if not generated_qs:
                print(f"  [{section}] No questions generated!")
                continue

            gen_texts    = [q.get("question_text", "") for q in generated_qs]
            actual_texts = [q.get("question_text", "") for q in actual_sec]

            # Semantic similarity
            sem = scorer.score(gen_texts, actual_texts)
            div = scorer.diversity(gen_texts)

            # Topic coverage
            gen_topics     = set(q.get("topic") for q in generated_qs)
            coverage       = len(gen_topics & actual_topics) / len(actual_topics) if actual_topics else 0.0

            # LLM-as-Judge (sample 3 per section to save quota)
            judge_scores = {}
            if GROQ_API_KEY:
                judge_scores = llm_judge_batch(generated_qs, groq, sample=3)

            sec_result = {
                "generated_count":  len(generated_qs),
                "actual_count":     len(actual_sec),
                "actual_topics":    sorted(actual_topics),
                "generated_topics": sorted(gen_topics),
                "topic_coverage":   round(coverage, 3),
                "semantic":         {k: round(v, 4) for k, v in sem.items()},
                "diversity":        round(div, 4),
                "llm_judge":        judge_scores,
            }
            year_result["sections"][section] = sec_result

            print(f"    Generated: {len(generated_qs)} | Coverage: {coverage:.1%} | "
                  f"SemanticSim: {sem['mean_max_sim']:.3f} | Diversity: {div:.3f}")
            if judge_scores:
                print(f"    LLM Judge: overall={judge_scores.get('overall',0):.1f}/5 "
                      f"correctness={judge_scores.get('correctness',0):.1f}/5")

        all_results.append(year_result)

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'Year':<6} {'Section':<22} {'Coverage':>10} {'Sem.Sim':>9} {'Diversity':>10} {'LLM Judge':>10}")
    print(f"{'─'*70}")
    for yr in all_results:
        for sec, res in yr["sections"].items():
            judge = res.get("llm_judge", {}).get("overall", "-")
            judge_str = f"{judge:.1f}" if isinstance(judge, float) else str(judge)
            print(f"{yr['test_year']:<6} {sec:<22} "
                  f"{res['topic_coverage']:>9.1%} "
                  f"{res['semantic']['mean_max_sim']:>9.3f} "
                  f"{res['diversity']:>10.3f} "
                  f"{judge_str:>10}")

    # Save JSON report
    report_path = EVAL_OUTPUT / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] Full report saved -> {report_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=TEST_YEARS)
    parser.add_argument("--qpt", type=int, default=QS_PER_TOPIC, help="Questions per topic")
    args = parser.parse_args()
    run_evaluation(test_years=args.years, questions_per_topic=args.qpt)
