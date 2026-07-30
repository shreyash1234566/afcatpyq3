"""
generate_questions_groq.py
==========================
Groq-powered AFCAT question generator using:
1. Topic-frequency forecasting (which topics to target)
2. Semantic RAG with MMR (which PYQ examples to provide)
3. Groq llama-3.3-70b-versatile (the LLM that generates)
4. Self-critique validation pass (LLM checks its own output)
5. Walk-forward filtering (no leakage -- only uses data < cutoff_year)
"""

import json
import re
import sys
import random
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 60

# Load API key from .env (same file the Flask server uses)
def _load_api_key() -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY=") and not line.endswith("gsk_your_api_key_here"):
                key = line.split("=", 1)[1].strip()
                if key and key.startswith("gsk_"):
                    return key
    import os
    key = os.environ.get("GROQ_API_KEY", "")
    if key and key.startswith("gsk_"):
        return key
    print("[CONFIG] No valid GROQ_API_KEY found!")
    print("[CONFIG] Please add it to e:\\afcatpyq3\\.env like:")
    print("[CONFIG]   GROQ_API_KEY=gsk_xxx...  (get one free at https://console.groq.com/keys)")
    return ""

GROQ_API_KEY = _load_api_key()

SAFE_REASONING_TOPICS = {
    "Verbal Analogy", "Verbal Classification", "Logical Reasoning",
    "Coding-Decoding", "Number/Letter Series", "Syllogism",
    "Direction Sense", "Blood Relations",
}

Q_JSON_PATH = ROOT / "data" / "processed" / "Q.json"
OUTPUT_PATH = ROOT / "output" / "generated_questions"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class GroqClient:
    def __init__(self, api_key=GROQ_API_KEY, model=GROQ_MODEL):
        self.api_key = api_key
        self.model   = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

    def chat(self, system_prompt, user_prompt, temperature=0.15, max_tokens=2048):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(GROQ_API_URL, headers=self.headers, json=payload, timeout=GROQ_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[GroqClient] API error: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# SEMANTIC RAG RETRIEVER
# ──────────────────────────────────────────────────────────────────────────────

class SemanticRAGRetriever:
    def __init__(self, all_questions, cutoff_year):
        self.cutoff_year = cutoff_year
        self.pool = [q for q in all_questions
                     if self._get_year(q.get("file_name","")) is not None
                     and self._get_year(q.get("file_name","")) < cutoff_year]
        self._model = None
        self._embeddings = None
        self._build_index()

    @staticmethod
    def _get_year(fn):
        m = re.search(r"(20\d\d)", fn or "")
        return int(m.group(1)) if m else None

    def _build_index(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  [RAG] Building index on {len(self.pool)} questions (year < {self.cutoff_year})...")
            self._model = SentenceTransformer("all-mpnet-base-v2")
            texts = [q.get("question_text","") for q in self.pool]
            self._embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            print(f"  [RAG] Index ready")
        except ImportError:
            print("  [RAG] WARNING: sentence_transformers not found -- using keyword fallback")

    def retrieve_mmr(self, topic, section, k=5, lambda_mult=0.6):
        candidates = [(i, q) for i, q in enumerate(self.pool)
                      if q.get("section") == section and q.get("topic") == topic
                      and q.get("question_text","").strip()]
        if not candidates:
            candidates = [(i, q) for i, q in enumerate(self.pool)
                          if q.get("section") == section and q.get("question_text","").strip()]
        if not candidates:
            return []

        if self._model is None or self._embeddings is None:
            random.shuffle(candidates)
            return [q for _, q in candidates[:k]]

        idxs = [i for i, _ in candidates]
        cand_embs = self._embeddings[idxs]
        query_emb = self._model.encode([f"AFCAT {topic} {section} question"], normalize_embeddings=True)

        selected, remaining = [], list(range(len(idxs)))
        for _ in range(min(k, len(remaining))):
            if not remaining:
                break
            if not selected:
                sims = cand_embs[remaining] @ query_emb[0]
                best = remaining[int(np.argmax(sims))]
            else:
                rel_sims = cand_embs[remaining] @ query_emb[0]
                div_sims = (cand_embs[remaining] @ cand_embs[selected].T).max(axis=1)
                scores   = lambda_mult * rel_sims - (1 - lambda_mult) * div_sims
                best     = remaining[int(np.argmax(scores))]
            selected.append(best)
            remaining.remove(best)

        return [candidates[i][1] for i in selected]


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior examiner who sets questions for the AFCAT (Air Force Common Admission Test), India.
You always respond with ONLY valid JSON and nothing else.
Your questions must be original, crystal-clear, and at the correct difficulty level.
Every distractor (wrong option) must be plausible but unambiguously wrong to a well-prepared student.
Include a concise step-by-step explanation for the correct answer."""

CRITIQUE_SYSTEM = """You are a strict quality-checker for competitive exam questions. You respond ONLY with valid JSON."""


def _fmt_examples(examples):
    if not examples:
        return "No examples available."
    parts = []
    for i, q in enumerate(examples, 1):
        opts = q.get("choices") or q.get("options") or []
        if opts and isinstance(opts[0], dict):
            opt_str = "\n".join(f"  ({o.get('key','?')}) {o.get('text','')}" for o in opts)
        else:
            opt_str = "\n".join(f"  {o}" for o in opts)
        yr = re.search(r"(20\d\d)", q.get("file_name",""))
        parts.append(f"[Example {i} - AFCAT {yr.group(1) if yr else '?'}]\n{q['question_text']}\n{opt_str}")
    return "\n\n".join(parts)


def _json_schema(topic, section, difficulty):
    return f"""{{
  "questions": [
    {{
      "question_text": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation": "...",
      "topic": "{topic}",
      "section": "{section}",
      "difficulty": "{difficulty}"
    }}
  ]
}}"""


def build_prompt_verbal(topic, examples, count, difficulty):
    return f"""Generate {count} new AFCAT Verbal Ability MCQs on topic: "{topic}". Difficulty: {difficulty.upper()}

Real past questions for style reference:
{_fmt_examples(examples)}

REQUIREMENTS:
- Test a specific English word / phrase / grammar rule
- 4 options (A-D); distractors must be plausible near-words
- Do NOT reuse any example above; use formal exam English

Return ONLY this JSON (no markdown):
{_json_schema(topic, 'Verbal Ability', difficulty)}"""


def build_prompt_numerical(topic, examples, count, difficulty):
    return f"""Generate {count} new AFCAT Numerical Ability MCQs on topic: "{topic}". Difficulty: {difficulty.upper()}

Real past questions (use as templates -- CHANGE the numbers/context):
{_fmt_examples(examples)}

REQUIREMENTS:
- Indian context (Rs., km, Indian names/cities)
- Verify your own arithmetic -- all calculations must be correct
- Distractors = common calculation mistakes
- Show full working in explanation

Return ONLY this JSON:
{_json_schema(topic, 'Numerical Ability', difficulty)}"""


def build_prompt_reasoning(topic, examples, count, difficulty):
    return f"""Generate {count} new AFCAT Reasoning MCQs on topic: "{topic}". Difficulty: {difficulty.upper()}

Real past questions for style reference:
{_fmt_examples(examples)}

REQUIREMENTS:
- TEXT-ONLY reasoning (no images or diagrams)
- For analogies: use a consistent logical relationship
- For coding: use one consistent substitution rule
- Distractors follow the same pattern but with a subtle error
- Do NOT reuse any example above

Return ONLY this JSON:
{_json_schema(topic, 'Reasoning', difficulty)}"""


def build_prompt_gk(topic, examples, count, difficulty):
    return f"""Generate {count} new AFCAT General Awareness MCQs on topic: "{topic}". Difficulty: {difficulty.upper()}

Real past questions for style reference:
{_fmt_examples(examples)}

REQUIREMENTS:
- Only include VERIFIED, established facts -- no current events that may be outdated
- Defence: IAF history, aircraft, ranks, milestones
- Science: physics/chemistry/biology principles
- Sports: well-known records and Indian achievements
- Distractors must be from the same category as the answer
- CRITICAL: only assert facts you are 100% certain about

Return ONLY this JSON:
{_json_schema(topic, 'General Awareness', difficulty)}"""


PROMPT_BUILDERS = {
    "Verbal Ability":    build_prompt_verbal,
    "Numerical Ability": build_prompt_numerical,
    "Reasoning":         build_prompt_reasoning,
    "General Awareness": build_prompt_gk,
}


def build_critique_prompt(q):
    return f"""Review this AFCAT MCQ:
{json.dumps(q, indent=2)}

Check:
1. Is correct_answer actually correct? (re-derive math if needed)
2. Are distractors plausible but clearly wrong?
3. Is the question unambiguous?

Return ONLY:
{{"is_valid": true, "issues": [], "corrected_answer": null}}"""


# ──────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class AFCATQuestionGenerator:

    def __init__(self, cutoff_year=2026, use_critique=True):
        self.cutoff_year = cutoff_year
        self.use_critique = use_critique
        self.groq = GroqClient()
        print(f"\n[Generator] Loading Q.json...")
        with open(Q_JSON_PATH, "r", encoding="utf-8") as f:
            self.all_questions = json.load(f)
        print(f"[Generator] {len(self.all_questions)} questions loaded")
        self.rag = SemanticRAGRetriever(self.all_questions, cutoff_year)

    def _parse_response(self, raw):
        if not raw:
            return []
        try:
            data = json.loads(raw)
            return data.get("questions", [])
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                try:
                    return json.loads(m.group()).get("questions", [])
                except Exception:
                    pass
        print(f"  [Parse] Failed to parse response")
        return []

    def _validate_question(self, q):
        return (
            isinstance(q.get("question_text"), str) and len(q["question_text"]) > 15
            and isinstance(q.get("options"), dict) and len(q["options"]) == 4
            and q.get("correct_answer") in {"A", "B", "C", "D"}
        )

    def _critique(self, q):
        raw = self.groq.chat(
            system_prompt=CRITIQUE_SYSTEM,
            user_prompt=build_critique_prompt(q),
            temperature=0.0,
            max_tokens=512,
        )
        if not raw:
            return q
        try:
            result = json.loads(raw)
            if not result.get("is_valid", True):
                print(f"    [Critique] Flagged: {result.get('issues')}")
                return None
            corr = result.get("corrected_answer")
            if corr and corr in "ABCD":
                q["correct_answer"] = corr
        except Exception:
            pass
        return q

    def generate_topic(self, topic, section, count=3, difficulty="medium"):
        if section == "Reasoning" and topic not in SAFE_REASONING_TOPICS:
            print(f"  [Skip] {topic} -- requires figures")
            return []

        print(f"\n  -> Generating {count}x {difficulty} | {section} | {topic}")
        examples = self.rag.retrieve_mmr(topic, section, k=5)
        print(f"    RAG: {len(examples)} examples")

        builder = PROMPT_BUILDERS.get(section, build_prompt_gk)
        raw = self.groq.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=builder(topic, examples, count, difficulty),
            temperature=0.15,
            max_tokens=2048,
        )

        questions = self._parse_response(raw)
        valid = [q for q in questions if self._validate_question(q)]

        if self.use_critique:
            critiqued = []
            for q in valid:
                r = self._critique(q)
                if r is not None:
                    critiqued.append(r)
            valid = critiqued
            print(f"    After critique: {len(valid)} valid")

        for q in valid:
            q["source"]             = f"groq-{GROQ_MODEL}"
            q["cutoff_year"]        = self.cutoff_year
            q["rag_examples_count"] = len(examples)
        return valid

    def generate_batch(self, topic_plan):
        all_results = defaultdict(list)
        for topic, section, count, difficulty in topic_plan:
            qs = self.generate_topic(topic, section, count, difficulty)
            all_results[section].extend(qs)
        total = sum(len(v) for v in all_results.values())
        print(f"\n[Generator] Total generated: {total}")
        return dict(all_results)

    def save(self, results, tag="2026"):
        flat = [q for qs in results.values() for q in qs]
        out_path = OUTPUT_PATH / f"generated_{tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)
        print(f"[Generator] Saved {len(flat)} questions -> {out_path}")
        return out_path


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT TOPIC PLAN  (topic, section, count, difficulty)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TOPIC_PLAN = [
    # Verbal Ability
    ("Synonyms",               "Verbal Ability",    4, "medium"),
    ("Antonyms",               "Verbal Ability",    4, "medium"),
    ("Idioms & Phrases",       "Verbal Ability",    4, "medium"),
    ("Error Detection",        "Verbal Ability",    3, "medium"),
    ("Sentence Completion",    "Verbal Ability",    3, "medium"),
    ("One Word Substitution",  "Verbal Ability",    2, "medium"),
    ("Cloze Test",             "Verbal Ability",    2, "medium"),
    # Numerical Ability
    ("Average",                "Numerical Ability", 3, "medium"),
    ("Percentage",             "Numerical Ability", 3, "medium"),
    ("Profit and Loss",        "Numerical Ability", 3, "medium"),
    ("Simple Interest",        "Numerical Ability", 2, "medium"),
    ("Ratio and Proportion",   "Numerical Ability", 2, "medium"),
    ("Time and Work",          "Numerical Ability", 2, "medium"),
    ("Compound Interest",      "Numerical Ability", 2, "medium"),
    # Reasoning (text-only)
    ("Verbal Analogy",         "Reasoning",         5, "medium"),
    ("Verbal Classification",  "Reasoning",         4, "medium"),
    ("Logical Reasoning",      "Reasoning",         3, "medium"),
    ("Coding-Decoding",        "Reasoning",         3, "medium"),
    ("Number/Letter Series",   "Reasoning",         3, "medium"),
    ("Syllogism",              "Reasoning",         2, "medium"),
    # General Awareness
    ("Defence",                "General Awareness", 3, "medium"),
    ("Science",                "General Awareness", 3, "medium"),
    ("Sports",                 "General Awareness", 2, "medium"),
    ("Polity",                 "General Awareness", 2, "medium"),
    ("Modern History",         "General Awareness", 2, "medium"),
    ("World Geography",        "General Awareness", 2, "medium"),
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-year", type=int, default=2026)
    parser.add_argument("--tag", type=str, default="2026")
    parser.add_argument("--no-critique", action="store_true")
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--section", type=str, default=None)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    gen = AFCATQuestionGenerator(cutoff_year=args.cutoff_year, use_critique=not args.no_critique)

    if args.topic:
        qs = gen.generate_topic(args.topic, args.section or "Verbal Ability", args.count)
        print(json.dumps(qs, indent=2, ensure_ascii=False))
    else:
        results = gen.generate_batch(DEFAULT_TOPIC_PLAN)
        gen.save(results, tag=args.tag)
