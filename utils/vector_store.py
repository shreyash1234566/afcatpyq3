"""
utils/vector_store.py
=====================
Persistent Vector and Structural Pattern Store for AFCAT Exam Questions.

Combines:
  1. SQLite FTS5 for BM25 lexical token & phrase search.
  2. Scikit-Learn TF-IDF for dense n-gram semantic similarity.
  3. Exponential recency weighting prioritizing 2026 (Jan & Aug) and 2025 exemplars.
  4. Structural 5-word question stem skeleton extraction.

Indexed over all 3,161 AFCAT questions (2011–2026) in data/processed/Q_clean.json.
Stores persistent index in data/afcat_pattern_store.db.
"""

from __future__ import annotations
import json
import re
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "afcat_pattern_store.db"
TFIDF_PATH = ROOT / "data" / "afcat_tfidf_model.pkl"
Q_CLEAN_PATH = ROOT / "data" / "processed" / "Q_clean.json"


def _extract_year(fn: str, test_name: str = "") -> int:
    m = re.search(r"(20\d\d)", f"{fn} {test_name}")
    return int(m.group(1)) if m else 2011


def _extract_template(text: str) -> str:
    """Extract first 5-6 words normalized as structural question template."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    words = clean.split()
    if len(words) >= 4:
        return " ".join(words[:5])
    return clean.strip()


class VectorPatternStore:
    def __init__(self, db_path: Path = DB_PATH, tfidf_path: Path = TFIDF_PATH):
        self.db_path = Path(db_path)
        self.tfidf_path = Path(tfidf_path)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.doc_ids: List[int] = []

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create tables and FTS5 index if not exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT,
                    test_name TEXT,
                    year INTEGER,
                    section TEXT,
                    topic TEXT,
                    micro_topic TEXT,
                    question_number INTEGER,
                    question_text TEXT,
                    stem_template TEXT,
                    options_json TEXT,
                    correct_answer TEXT,
                    explanation TEXT,
                    difficulty TEXT,
                    has_figure INTEGER,
                    image_path TEXT,
                    passage TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sec_top ON questions(section, topic);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON questions(year);")

            # Check if FTS table exists
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='questions_fts';")
            if not cur.fetchone():
                conn.execute("""
                    CREATE VIRTUAL TABLE questions_fts USING fts5(
                        question_id UNINDEXED,
                        question_text,
                        stem_template,
                        topic,
                        micro_topic,
                        section
                    );
                """)
            conn.commit()

    def build_from_clean_data(self, q_clean_path: Path = Q_CLEAN_PATH, force: bool = False):
        """Index all questions from Q_clean.json into SQLite + FTS5 and fit TF-IDF."""
        self.init_db()

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM questions;")
            existing_count = cur.fetchone()[0]

            if existing_count > 0 and not force:
                print(f"[VectorPatternStore] Database already indexed with {existing_count} questions.")
                self._load_tfidf()
                return

            print(f"[VectorPatternStore] Building store from {q_clean_path}...")
            conn.execute("DELETE FROM questions;")
            conn.execute("DELETE FROM questions_fts;")

            with open(q_clean_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

            corpus = []
            records = []
            for q in questions:
                txt = q.get("question_text", "").strip()
                fn = q.get("file_name", "")
                tn = q.get("test_name", "")
                yr = _extract_year(fn, tn)
                tmpl = _extract_template(txt)
                sec = q.get("section", "")
                top = q.get("topic", "")
                micro = q.get("micro_topic", top)
                opts = q.get("choices") or q.get("options") or []
                opts_json = json.dumps(opts)
                corr = q.get("correct_answer", "")
                expl = q.get("explanation", "")
                diff = q.get("difficulty", "Medium")
                fig = 1 if q.get("has_figure") else 0
                img_raw = q.get("image_path") or q.get("all_images") or ""
                img = ", ".join(img_raw) if isinstance(img_raw, list) else str(img_raw)
                pas_raw = q.get("passage") or ""
                pas = str(pas_raw) if pas_raw else ""

                cur.execute("""
                    INSERT INTO questions (
                        file_name, test_name, year, section, topic, micro_topic,
                        question_number, question_text, stem_template, options_json,
                        correct_answer, explanation, difficulty, has_figure, image_path, passage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fn, tn, yr, sec, top, micro, q.get("question_number", 0),
                      txt, tmpl, opts_json, corr, expl, diff, fig, img, pas))
                qid = cur.lastrowid

                cur.execute("""
                    INSERT INTO questions_fts (
                        question_id, question_text, stem_template, topic, micro_topic, section
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (qid, txt, tmpl, top, micro, sec))

                corpus.append(f"{top} {micro} {tmpl} {txt}")
                records.append(qid)

            conn.commit()
            print(f"[VectorPatternStore] Successfully indexed {len(records)} questions in SQLite + FTS5.")

        # Train TF-IDF vectorizer
        print("[VectorPatternStore] Fitting TF-IDF n-gram vectorizer...")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.doc_ids = records

        with open(self.tfidf_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "doc_ids": self.doc_ids
            }, f)
        print(f"[VectorPatternStore] Saved TF-IDF matrix to {self.tfidf_path}.")

    def _load_tfidf(self):
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            return
        if self.tfidf_path.exists():
            with open(self.tfidf_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["tfidf_matrix"]
                self.doc_ids = data["doc_ids"]
        else:
            self.build_from_clean_data(force=True)

    def query_patterns(
        self,
        section: str,
        topic: str,
        micro_topic: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining:
          1. Topic & micro-topic candidate retrieval.
          2. TF-IDF cosine similarity against target query/pattern.
          3. Recency weighting favoring 2026 (both shifts) and 2025.
        """
        self._load_tfidf()

        with self._get_conn() as conn:
            cur = conn.cursor()
            if micro_topic:
                cur.execute("""
                    SELECT * FROM questions
                    WHERE section = ? AND topic = ? AND micro_topic = ?
                """, (section, topic, micro_topic))
                candidates = cur.fetchall()
            else:
                candidates = []

            # If fewer than top_k, fallback to broad topic
            if len(candidates) < top_k:
                cur.execute("""
                    SELECT * FROM questions
                    WHERE section = ? AND topic = ?
                """, (section, topic))
                candidates = cur.fetchall()

        if not candidates:
            return []

        cand_dict_list = [dict(c) for c in candidates]
        target_str = f"{topic} {micro_topic or ''} {query_text or ''}".strip()
        target_vec = self.vectorizer.transform([target_str])

        cand_ids = [c["id"] for c in cand_dict_list]
        id_to_idx = {qid: idx for idx, qid in enumerate(self.doc_ids)}
        valid_indices = [id_to_idx[qid] for qid in cand_ids if qid in id_to_idx]

        if valid_indices:
            cand_tfidf = self.tfidf_matrix[valid_indices]
            sims = cosine_similarity(target_vec, cand_tfidf).flatten()
        else:
            sims = np.zeros(len(cand_dict_list))

        # Score with recency boost
        scored = []
        for i, c in enumerate(cand_dict_list):
            sim = sims[i] if i < len(sims) else 0.0
            yr = c.get("year", 2011)
            # Recency weighting: 2026 -> 2.5, 2025 -> 1.8, 2024 -> 1.4, older -> 1.0
            if yr >= 2026:
                rec_w = 2.5
            elif yr == 2025:
                rec_w = 1.8
            elif yr == 2024:
                rec_w = 1.4
            else:
                rec_w = 1.0

            # Match micro-topic bonus
            micro_match = 1.5 if (micro_topic and c.get("micro_topic") == micro_topic) else 1.0

            final_score = (0.5 * sim + 0.5) * rec_w * micro_match
            c_copy = dict(c)
            c_copy["score"] = float(final_score)
            c_copy["similarity"] = float(sim)
            try:
                c_copy["options"] = json.loads(c["options_json"])
            except Exception:
                c_copy["options"] = {}
            scored.append(c_copy)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_top_templates(self, topic: str, micro_topic: Optional[str] = None, n: int = 3) -> List[Dict[str, Any]]:
        """Get the most frequent structural 5-word skeletons for a topic/micro-topic."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            if micro_topic:
                cur.execute("""
                    SELECT stem_template, question_text, COUNT(*) as freq
                    FROM questions
                    WHERE topic = ? AND micro_topic = ?
                    GROUP BY stem_template
                    ORDER BY freq DESC, year DESC
                    LIMIT ?
                """, (topic, micro_topic, n))
            else:
                cur.execute("""
                    SELECT stem_template, question_text, COUNT(*) as freq
                    FROM questions
                    WHERE topic = ?
                    GROUP BY stem_template
                    ORDER BY freq DESC, year DESC
                    LIMIT ?
                """, (topic, n))
            rows = cur.fetchall()
            return [{"template": r["stem_template"], "example": r["question_text"], "frequency": r["freq"]} for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM questions;")
            total = cur.fetchone()[0]

            cur.execute("SELECT section, COUNT(*) FROM questions GROUP BY section;")
            by_sec = dict(cur.fetchall())

            cur.execute("SELECT year, COUNT(*) FROM questions GROUP BY year ORDER BY year;")
            by_yr = dict(cur.fetchall())

            return {
                "total_questions": total,
                "by_section": by_sec,
                "by_year": by_yr
            }


if __name__ == "__main__":
    store = VectorPatternStore()
    store.build_from_clean_data(force=True)
    stats = store.get_stats()
    print("\nPattern Store Statistics:")
    print(json.dumps(stats, indent=2))

    # Test query
    sample = store.query_patterns("Numerical Ability", "Time, Speed & Dist.", "Boats & Streams", top_k=2)
    print("\nSample Query Results:")
    for s in sample:
        print(f"  [{s['year']}] {s['stem_template']} | Score: {s['score']:.2f} | Text: {s['question_text'][:80]}...")
