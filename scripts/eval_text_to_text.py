import json
import re
import sys
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

from scripts.generate_questions_groq import GroqClient, prompt_gk, prompt_verbal, prompt_reasoning, prompt_numerical

def _get_year(fn: str):
    m = re.search(r"(20\d\d)", fn or "")
    return int(m.group(1)) if m else None

def get_prompt(section, topic, examples, n):
    if section == "General Awareness":
        return prompt_gk(topic, examples, n)
    elif section == "Verbal Ability":
        return prompt_verbal(topic, examples, n)
    elif section == "Reasoning":
        return prompt_reasoning(topic, examples, n)
    elif section == "Numerical Ability":
        return prompt_numerical(topic, examples, n)
    return prompt_gk(topic, examples, n) # fallback

def main():
    if not SBERT_OK:
        print("SBERT not available.")
        return

    print("Loading data...")
    data = json.load(open(ROOT / "data" / "processed" / "Q.json", encoding="utf-8"))
    
    ctx_data = []
    test_2024 = defaultdict(lambda: defaultdict(list))
    
    for q in data:
        y = _get_year(q.get("file_name", ""))
        sec = q.get("section", "")
        top = q.get("topic", "")
        if not y or not sec or not top: continue
        
        # We only evaluate text-based predicting, so we skip image-heavy reasoning topics
        FIGURE_TOPICS = ["Venn Diagrams", "Non-Verbal Pattern", "Non-Verbal Series", 
                         "Spatial Ability", "Non-Verbal Classification", "Non-Verbal Analogy", 
                         "Dot Situation"]
        if sec == "Reasoning" and top in FIGURE_TOPICS:
            continue
            
        if y < 2024:
            ctx_data.append(q)
        elif y == 2024:
            test_2024[sec][top].append(q)

    print("Loading SBERT...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    groq = GroqClient()
    if not groq.ok:
        print("Groq key not found.")
        return

    print(f"\n--- FULL TEXT-TO-TEXT EVALUATION FOR 2024 (ALL SECTIONS) ---")
    
    section_sims = defaultdict(list)
    
    for sec, topics in test_2024.items():
        print(f"\n=========================================")
        print(f" {sec.upper()} ")
        print(f"=========================================")
        for topic, actual_qs in topics.items():
            print(f"\n[{topic}] (Actual 2024 Qs: {len(actual_qs)})")
            
            topic_ctx = [q for q in ctx_data if q.get("topic") == topic]
            random.shuffle(topic_ctx)
            examples = topic_ctx[:3]
            
            prompt = get_prompt(sec, topic, examples, len(actual_qs))
            sys_prompt = "You are an AFCAT examiner. Respond ONLY with valid JSON."
            resp = groq.chat(sys_prompt, prompt)
            
            generated = []
            if resp:
                try:
                    res_data = json.loads(resp)
                    if "questions" in res_data:
                        for obj in res_data["questions"]:
                            if "question" in obj:
                                generated.append(obj["question"])
                except Exception as e:
                    print(f"JSON Parse error: {e}")
                    
            if not generated:
                print("  -> Failed to generate prediction.")
                continue
                
            print("  -> Generated Predictions:")
            for g in generated: print(f"       - {g}")
            
            emb_gen = sbert.encode(generated, convert_to_numpy=True)
            emb_act = sbert.encode([a["question"] for a in actual_qs], convert_to_numpy=True)
            
            sims = np.dot(emb_gen, emb_act.T) / (
                np.linalg.norm(emb_gen, axis=1)[:, None] * np.linalg.norm(emb_act, axis=1)[None, :]
            )
            
            topic_avg_sim = np.mean(np.max(sims, axis=1))
            section_sims[sec].append(topic_avg_sim)
            print(f"  -> Semantic Similarity: {topic_avg_sim:.3f}")
            
    print(f"\n========================================================")
    print(f"FINAL EVALUATION RESULTS (2024):")
    overall = []
    for sec, sims in section_sims.items():
        if sims:
            sec_avg = np.mean(sims)
            overall.extend(sims)
            print(f"  {sec:<20}: {sec_avg:.3f}")
    
    if overall:
        print(f"\n  OVERALL ACCURACY  : {np.mean(overall):.3f}")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
