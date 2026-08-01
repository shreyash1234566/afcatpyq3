import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

from scripts.generate_questions_groq import GroqClient, prompt_gk

def _get_year(fn: str):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

def main():
    if not SBERT_OK:
        print("SBERT not available.")
        return

    print("Loading data...")
    data = json.load(open(ROOT / "data" / "processed" / "Q.json", encoding="utf-8"))
    
    # Split into Context (<2024) and Test (2024)
    ctx_data = []
    test_2024 = defaultdict(list)
    
    for q in data:
        y = _get_year(q.get("file_name", ""))
        sec = q.get("section", "")
        top = q.get("topic", "")
        if not y or sec != "General Awareness": continue
        
        if y < 2024:
            ctx_data.append(q)
        elif y == 2024:
            test_2024[top].append(q)

    # Prepare SBERT
    print("Loading SBERT...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    groq = GroqClient()
    if not groq.ok:
        print("Groq key not found.")
        return

    print(f"\n--- EVALUATING TEXT-TO-TEXT PREDICTION FOR 2024 GENERAL AWARENESS ---")
    print(f"Context questions (2014-2023): {len(ctx_data)}")
    print(f"Test topics (2024): {len(test_2024)}")
    
    all_sims = []
    
    for topic, actual_qs in test_2024.items():
        print(f"\n[{topic}] (Actual 2024 Qs: {len(actual_qs)})")
        
        # 1. Get few-shot context from <2024
        topic_ctx = [q for q in ctx_data if q.get("topic") == topic]
        # just pick top 3 at random for prompt
        import random
        random.shuffle(topic_ctx)
        examples = topic_ctx[:3]
        
        # 2. Predict (Generate) N questions via Groq
        prompt = prompt_gk(topic, examples, len(actual_qs))
        sys_prompt = "You are an AFCAT examiner. Respond ONLY with valid JSON."
        resp = groq.chat(sys_prompt, prompt)
        
        generated = []
        if resp:
            try:
                # Groq returns JSON object with a "questions" key
                data = json.loads(resp)
                if "questions" in data:
                    for obj in data["questions"]:
                        if "question" in obj:
                            generated.append(obj["question"])
            except Exception as e:
                print(f"JSON Parse error: {e}")
                
        if not generated:
            print("  -> Failed to generate prediction.")
            continue
            
        print("  -> Generated Predictions:")
        for g in generated: print(f"       - {g}")
        print("  -> Actual 2024 Questions:")
        for a in actual_qs: print(f"       - {a['question']}")
        
        # 3. Compute semantic similarity
        emb_gen = sbert.encode(generated, convert_to_numpy=True)
        emb_act = sbert.encode([a["question"] for a in actual_qs], convert_to_numpy=True)
        
        # for each generated question, find the max similarity to an actual question
        # dot product works since we normalize, or cosine_similarity
        sims = np.dot(emb_gen, emb_act.T) / (
            np.linalg.norm(emb_gen, axis=1)[:, None] * np.linalg.norm(emb_act, axis=1)[None, :]
        )
        
        # we take the max similarity for each generated question
        topic_avg_sim = np.mean(np.max(sims, axis=1))
        all_sims.append(topic_avg_sim)
        print(f"  -> Semantic Similarity: {topic_avg_sim:.3f}")
        
    print(f"\n========================================================")
    print(f"OVERALL TEXT-TO-TEXT PREDICTION ACCURACY (2024 GK): {np.mean(all_sims):.3f}")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
