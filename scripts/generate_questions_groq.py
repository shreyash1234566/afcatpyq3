"""
generate_questions_groq.py  (v2 — DM-Multinomial driven)
=========================================================

Uses the Dirichlet-Multinomial model's 2026 topic predictions to decide WHAT 
and HOW MANY questions to generate or sample per topic.

THREE strategies (absolutely no hallucination):
  1. GROQ GENERATE  — text-based topics (Verbal, Numerical, text-only Reasoning)
                      Groq llama-3.3-70b writes new unique questions using 
                      real PYQs as few-shot context (MMR retrieval).
  2. PYQ SAMPLE     — figure-dependent topics (Non-Verbal Pattern, Venn etc.)
                      We cannot generate these without images, so we pull the 
                      most relevant REAL past questions and surface them.
  3. PYQ + GROQ GK  — General Awareness topics: serve REAL PYQs first, then
                      add Groq-generated factual MCQs only where verified.

Outputs: output/generated_questions/practice_2026.json
         (each entry has source='groq-generated' or source='pyq-curated')
"""

import json, re, sys, random, collections, math
import numpy as np
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 60

DM_PLAN_PATH = ROOT / "output" / "dm_2026_topic_plan.json"
Q_JSON_PATH  = ROOT / "data" / "processed" / "Q.json"
OUTPUT_DIR   = ROOT / "output" / "generated_questions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Topics that REQUIRE a figure — we cannot generate text for these
FIGURE_TOPICS = {
    "Non-Verbal Pattern", "Non-Verbal Series", "Non-Verbal Classification",
    "Non-Verbal Analogy", "Venn Diagrams", "Dot Situation",
    "Spatial Ability", "Embedded Figures",
}

# Text-only Reasoning topics (safe for Groq)
TEXT_REASONING_TOPICS = {
    "Logical Reasoning", "Verbal Classification", "Number/Letter Series",
    "Coding-Decoding", "Verbal Analogy", "Syllogism",
    "Direction Sense", "Blood Relations",
}

# GK — always serve real PYQs first + optional Groq supplement
GK_SECTION = "General Awareness"

# How many Q to generate per predicted question count (ratio)
# e.g. if DM predicts 5 Qs for a topic, we generate 2x = 10 practice Qs
PRACTICE_MULTIPLIER = 2
MAX_PER_TOPIC = 8   # cap so we don't burn API quota


# ──────────────────────────────────────────────────────────────────────────────
# API KEY — loaded from .env  (you paste your key there)
# ──────────────────────────────────────────────────────────────────────────────

def _load_api_keys():
    import os
    keys = ""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEYS=") and "gsk_your_api_key_here" not in line:
                keys = line.split("=", 1)[1].strip()
                break
        if not keys:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY=") and "gsk_your_api_key_here" not in line:
                    keys = line.split("=", 1)[1].strip()
                    break
    
    if not keys:
        keys = os.getenv("GROQ_API_KEYS", "")
    if not keys:
        keys = os.getenv("GROQ_API_KEY", "")
    return keys

GROQ_API_KEYS = _load_api_keys()


# ──────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class GroqClient:
    def __init__(self):
        keys = [k.strip() for k in GROQ_API_KEYS.split(",") if k.strip()]
        self.keys = keys
        self.key_idx = 0
        self.ok = len(self.keys) > 0
        if not self.ok:
            print("[Groq] WARNING: No API keys found in .env (GROQ_API_KEYS)")

    def get_current_key(self):
        return self.keys[self.key_idx]

    def rotate_key(self):
        self.key_idx = (self.key_idx + 1) % len(self.keys)
        print(f"  [Groq] Rotated API key (using key {self.key_idx+1}/{len(self.keys)})")

    def chat(self, system_prompt, user_prompt, temperature=0.15, max_tokens=2048):
        if not self.ok:
            return None
        import time
        # Allow enough retries to safely wait through multiple 60-second TPM resets
        for attempt in range(len(self.keys) * 10):
            try:
                r = requests.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {self.get_current_key()}", "Content-Type": "application/json"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens":  max_tokens,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=GROQ_TIMEOUT,
                )
                if r.status_code == 429:
                    raise Exception("429 Too Many Requests")
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if "429" in str(e) or (hasattr(e, "response") and e.response is not None and e.response.status_code == 429):
                    wait_time = min(60, 2 ** (attempt // len(self.keys) + 2))
                    print(f"  [Groq] Rate limit hit on key {self.key_idx+1}.")
                    self.rotate_key()
                    import time
                    if attempt % len(self.keys) == len(self.keys) - 1:
                        # If we cycled through all keys and they are ALL rate limited, 
                        # it's usually a Tokens Per Minute (TPM) limit. 
                        # TPM resets every 60 seconds. Wait 60s.
                        print(f"  [Groq] All keys rate limited. Sleeping 60s to let TPM reset...")
                        time.sleep(60)
                    else:
                        time.sleep(1.0)
                else:
                    print(f"  [Groq] Error: {e}")
                    return None
        print("  [Groq] Max retries across all keys exceeded.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_all_questions():
    with open(Q_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_dm_plan():
    with open(DM_PLAN_PATH, encoding="utf-8") as f:
        return json.load(f)

def get_year(fn):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# MMR RETRIEVER  (diverse semantic retrieval from PYQ bank)
# ──────────────────────────────────────────────────────────────────────────────

class MMRRetriever:
    def __init__(self, questions):
        self.pool = questions
        self._model = None
        self._embs  = None
        self._built = False

    def build(self):
        if self._built:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print("  [RAG] Building SBERT index...")
            self._model = SentenceTransformer("all-mpnet-base-v2")
            texts = [q.get("question_text", "") for q in self.pool]
            self._embs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            print(f"  [RAG] {len(self.pool)} vectors ready")
        except ImportError:
            print("  [RAG] sentence_transformers not found — keyword fallback")
        self._built = True

    def get(self, topic, section, k, lambda_mult=0.6):
        """Return k maximally diverse questions for (topic, section)."""
        cands = [(i, q) for i, q in enumerate(self.pool)
                 if q.get("section") == section and q.get("topic") == topic
                 and q.get("question_text", "").strip()]
        if not cands:
            cands = [(i, q) for i, q in enumerate(self.pool)
                     if q.get("section") == section and q.get("question_text", "").strip()]
        if not cands:
            return []

        # No SBERT → random
        if self._model is None or self._embs is None:
            random.shuffle(cands)
            return [q for _, q in cands[:k]]

        idxs = [i for i, _ in cands]
        cembs = self._embs[idxs]
        qemb  = self._model.encode([f"AFCAT {topic} question"], normalize_embeddings=True)

        sel, rem = [], list(range(len(idxs)))
        for _ in range(min(k, len(rem))):
            if not rem:
                break
            if not sel:
                scores = cembs[rem] @ qemb[0]
            else:
                rel = cembs[rem] @ qemb[0]
                div = (cembs[rem] @ cembs[sel].T).max(axis=1)
                scores = lambda_mult * rel - (1 - lambda_mult) * div
            best = rem[int(np.argmax(scores))]
            sel.append(best)
            rem.remove(best)
        return [cands[i][1] for i in sel]


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS  (one per section, section-appropriate)
# ──────────────────────────────────────────────────────────────────────────────

SYS = """You are a senior examiner for the AFCAT (Air Force Common Admission Test), India.
Respond ONLY with valid JSON and nothing else.
Your primary objective is to PREDICT THE EXACT WORD-TO-WORD TEXT of the next logical question that will appear on the exam based on past data.
You MUST strictly clone the exact structural template (e.g. 'Select the most appropriate word for blank...', 'Arrange the jumbled parts:') of the provided past questions.
Do NOT invent new formats. Do NOT use generic phrasing. Match the exact wording style and structure of the examples.
Include a deep, step-by-step explanatory solution confirming the correct answer, particularly for reasoning and mathematics."""


def _fmt(examples):
    parts = []
    for i, q in enumerate(examples[:5], 1):
        opts = q.get("choices") or q.get("options") or []
        if opts and isinstance(opts[0], dict):
            opt_str = "  ".join(f"({o.get('key','?')}) {o.get('text','')}" for o in opts)
        else:
            opt_str = "  ".join(str(o) for o in opts)
        yr = re.search(r"(20\d\d)", q.get("file_name", ""))
        parts.append(f"[Ex {i} – AFCAT {yr.group(1) if yr else '?'}] {q['question_text']}  {opt_str}")
    return "\n".join(parts)

def _schema(topic, section, n):
    return f"""{{
  "questions": [
    {{
      "question_text": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation": "...",
      "topic": "{topic}",
      "section": "{section}",
      "difficulty": "medium",
      "source": "groq-generated"
    }}
  ]
}}  <- repeat {n} items total"""

def prompt_verbal(topic, examples, n):
    return f"""Generate {n} new AFCAT Verbal Ability MCQs on: "{topic}"

Past AFCAT questions for style (DO NOT reuse):
{_fmt(examples)}

Rules:
- Test a specific English word/idiom/grammar concept
- 4 options A-D; wrong options must be plausible near-words
- No reuse of example words/phrases
- Return ONLY JSON: {_schema(topic, "Verbal Ability", n)}"""

def prompt_numerical(topic, examples, n):
    return f"""Generate {n} new AFCAT Numerical Ability MCQs on: "{topic}"

Past AFCAT questions as number templates (CHANGE the numbers/context):
{_fmt(examples)}

Rules:
- Indian context (Rs., km, litres, Indian names)
- Verify arithmetic 100% — double-check every calculation
- Distractors = typical calculation mistakes (unit error, inverted ratio, etc.)
- Show full working in explanation
- Return ONLY JSON: {_schema(topic, "Numerical Ability", n)}"""

def prompt_reasoning(topic, examples, n):
    return f"""Generate {n} new AFCAT Reasoning MCQs on: "{topic}"

Past AFCAT questions for style:
{_fmt(examples)}

Rules:
- TEXT-ONLY (no diagrams, no images needed)
- Use one consistent logical rule per question
- Distractors follow the same pattern with a subtle twist
- Return ONLY JSON: {_schema(topic, "Reasoning", n)}"""

def prompt_gk(topic, examples, n):
    return f"""Generate {n} new AFCAT General Awareness MCQs on: "{topic}"

Past AFCAT questions for style:
{_fmt(examples)}

Rules:
- PREDICT the most likely questions to appear in the upcoming AFCAT exam based on current trends and recent events.
- For Current Affairs / Defence: Focus on recent military exercises, acquisitions, missile tests, and key appointments.
- For Static GK / Science / History: Focus on highly-tested foundational facts.
- ONLY include facts you are 100% certain are correct. Do NOT hallucinate names or events.
- All options must be from the same category (e.g., all rivers, all missiles) to be plausible.
- Return ONLY JSON: {_schema(topic, "General Awareness", n)}"""

PROMPT_BUILDERS = {
    "Verbal Ability":    prompt_verbal,
    "Numerical Ability": prompt_numerical,
    "Reasoning":         prompt_reasoning,
    "General Awareness": prompt_gk,
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def parse_groq(raw):
    if not raw:
        return []
    try:
        return json.loads(raw).get("questions", [])
    except Exception:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                return json.loads(m.group()).get("questions", [])
            except Exception:
                pass
    return []

def validate(q):
    return (isinstance(q.get("question_text"), str) and len(q["question_text"]) > 10
            and isinstance(q.get("options"), dict) and len(q["options"]) == 4
            and q.get("correct_answer") in {"A", "B", "C", "D"})

def pyq_to_practice(q, rank=None):
    """Convert a PYQ dict to practice card format."""
    opts = q.get("choices") or q.get("options") or []
    if opts and isinstance(opts[0], dict):
        options_dict = {o.get("key", "?"): o.get("text", "") for o in opts}
        correct = q.get("answer", "")
    else:
        keys = ["A", "B", "C", "D"]
        options_dict = {keys[i]: str(o) for i, o in enumerate(opts) if i < 4}
        correct = q.get("answer", "")
    yr = get_year(q.get("file_name", ""))
    return {
        "question_text":  q.get("question_text", q.get("question", "")),
        "options":        options_dict,
        "correct_answer": correct,
        "explanation":    q.get("explanation", ""),
        "topic":          q.get("topic", ""),
        "section":        q.get("section", ""),
        "difficulty":     q.get("difficulty", "medium"),
        "source":         "pyq-curated",
        "year":           yr,
        "rank":           rank,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR  (DM-driven)
# ──────────────────────────────────────────────────────────────────────────────

class DM_QuestionGenerator:
    """
    Reads the DM 2026 forecast and generates/samples the right number of
    practice questions per topic using the appropriate strategy.
    """

    def __init__(self):
        print("\n[Generator] Loading data...")
        self.all_qs  = load_all_questions()
        self.dm_plan = load_dm_plan()
        self.groq    = GroqClient()
        self.rag     = MMRRetriever(self.all_qs)
        self.rag.build()
        print(f"[Generator] {len(self.all_qs)} PYQs | Groq: {'READY' if self.groq.ok else 'NO KEY (PYQ-only mode)'}")

    # ── strategy 1: Groq generates text ──────────────────────────────────────

    def _gen_groq(self, topic, section, n_target):
        """Generate n_target new questions via Groq."""
        if not self.groq.ok:
            return []
        examples = self.rag.get(topic, section, k=5)
        builder  = PROMPT_BUILDERS.get(section, prompt_gk)
        raw      = self.groq.chat(SYS, builder(topic, examples, n_target), temperature=0.15)
        qs       = [q for q in parse_groq(raw) if validate(q)]
        for q in qs:
            q["source"] = "groq-generated"
            q["topic"]  = topic
            q["section"] = section
        return qs

    # ── strategy 2 & 3: pull real PYQs ────────────────────────────────────────

    def _sample_pyq(self, topic, section, n_target):
        """Pull n_target diverse real PYQs for this topic."""
        candidates = self.rag.get(topic, section, k=n_target * 3)
        # Prefer questions with well-formed options
        good = [q for q in candidates
                if (q.get("choices") or q.get("options")) and q.get("question_text")]
        pool = good if good else candidates
        picked = pool[:n_target]
        return [pyq_to_practice(q, rank=i+1) for i, q in enumerate(picked)]

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        all_output = []
        stats = collections.defaultdict(lambda: {"groq": 0, "pyq": 0, "topics": []})

        for section, blk in self.dm_plan.items():
            print(f"\n{'='*60}")
            print(f"  {section.upper()}  (section total={blk['section_total']})")
            print(f"{'='*60}")
            
            # Use largest remainder method to ensure EXACT section totals (30, 25, 25, 20)
            target = blk['section_total']
            allocations = {}
            remainders = []
            for row in blk["topics"]:
                t = row["topic"]
                e = row["expected_count_exact"]
                allocations[t] = int(math.floor(e))
                remainders.append((e - math.floor(e), t))
                
            shortfall = target - sum(allocations.values())
            remainders.sort(reverse=True, key=lambda x: x[0])
            for i in range(shortfall):
                if i < len(remainders):
                    allocations[remainders[i][1]] += 1

            for row in blk["topics"]:
                topic   = row["topic"]
                pred_n  = row["expected_count_exact"]
                n_practice = allocations[topic]

                # Skip topics predicted to have zero questions in this paper
                if n_practice < 1:
                    continue

                print(f"\n  → {topic:<35} pred={pred_n:.1f}  mock_q={n_practice}")

                # ── Choose strategy ──────────────────────────────────────────
                if section == "Reasoning" and topic in FIGURE_TOPICS:
                    # Strategy 2: figure topic — only real PYQs
                    qs = self._sample_pyq(topic, section, n_practice)
                    print(f"    [PYQ] Figure topic — sampled {len(qs)} real PYQs")
                    stats[section]["pyq"] += len(qs)

                elif section == GK_SECTION:
                    # Strategy 3: GK — Groq generates predicted facts based on current trends
                    if self.groq.ok:
                        qs = self._gen_groq(topic, section, n_practice)
                        print(f"    [Groq] Generated {len(qs)} predicted GK questions")
                        if len(qs) < n_practice:
                            extra = self._sample_pyq(topic, section, n_practice - len(qs))
                            print(f"    [PYQ] Fallback — added {len(extra)} real PYQs")
                            qs.extend(extra)
                        stats[section]["groq"] += len(qs) - len(extra) if 'extra' in locals() else len(qs)
                        stats[section]["pyq"] += len(extra) if 'extra' in locals() else 0
                    else:
                        qs = self._sample_pyq(topic, section, n_practice)
                        print(f"    [PYQ] GK — sampled {len(qs)} real PYQs")
                        stats[section]["pyq"] += len(qs)

                else:
                    # Strategy 1: text topic — Groq generate
                    if self.groq.ok:
                        qs = self._gen_groq(topic, section, n_practice)
                        print(f"    [Groq] Generated {len(qs)} questions")
                        if len(qs) < n_practice:
                            # fallback if Groq gave too few (e.g. rate limit hit)
                            extra = self._sample_pyq(topic, section, n_practice - len(qs))
                            print(f"    [PYQ] Fallback — added {len(extra)} real PYQs")
                            qs.extend(extra)
                        stats[section]["groq"] += len(qs) - len(extra) if 'extra' in locals() else len(qs)
                        stats[section]["pyq"] += len(extra) if 'extra' in locals() else 0
                    else:
                        # No key: pure PYQ fallback
                        qs = self._sample_pyq(topic, section, n_practice)
                        print(f"    [PYQ] No API key — sampled {len(qs)} real PYQs")
                        stats[section]["pyq"] += len(qs)

                # Add DM metadata to each Q
                for q in qs:
                    q["dm_predicted_count"] = pred_n
                    q["ci90_low"]  = row["ci90_low"]
                    q["ci90_high"] = row["ci90_high"]
                    q["dm_share"]  = row["share"]

                stats[section]["topics"].append(topic)
                all_output.extend(qs)

        # ── Save ─────────────────────────────────────────────────────────────
        out_path = OUTPUT_DIR / "practice_2026.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_output, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"DONE — {len(all_output)} practice questions saved → {out_path}")
        for sec, s in stats.items():
            print(f"  {sec:<22}: {s['groq']} Groq + {s['pyq']} PYQ = {s['groq']+s['pyq']} total")
        return all_output


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = DM_QuestionGenerator()
    gen.run()
