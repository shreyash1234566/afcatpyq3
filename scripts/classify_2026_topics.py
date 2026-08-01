"""
Classify extracted 2026 questions into Sections and Canonical Topics.
"""
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_questions_groq import GroqClient

RAW_JSON = ROOT / "data" / "processed" / "Q_2026_extracted_raw.json"
OUT_JSON = ROOT / "data" / "processed" / "Q_2026_classified.json"
TOPIC_MAP = ROOT / "data" / "topic_map.json"

def main():
    questions = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    topic_map = json.loads(TOPIC_MAP.read_text(encoding="utf-8"))
    
    canonical_topics = set(topic_map.values())
    topics_list_str = "\n".join(f"- {t}" for t in sorted(canonical_topics))
    
    groq = GroqClient()
    
    SYS = "You are an expert at classifying AFCAT exam questions into predefined topics. Respond ONLY with valid JSON."
    
    classified = []
    
    # Process in batches of 10
    batch_size = 10
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        q_text_block = "\n".join([f"[{j}]: {q['question_text']}" for j, q in enumerate(batch)])
        
        prompt = f"""Classify the following {len(batch)} AFCAT questions into one of the exact canonical topics below.
        
CANONICAL TOPICS:
{topics_list_str}

SECTIONS: Verbal Ability, General Awareness, Reasoning, Numerical Ability

QUESTIONS:
{q_text_block}

Respond strictly with this JSON format:
{{
  "classifications": [
    {{
      "index": 0,
      "section": "...",
      "topic": "..."
    }}
  ]
}}
"""
        
        print(f"Classifying batch {i//batch_size + 1}/{(len(questions)+batch_size-1)//batch_size}...")
        
        retries = 3
        while retries > 0:
            resp = groq.chat(SYS, prompt)
            try:
                if not resp: raise ValueError("Empty response")
                res_data = json.loads(resp)
                
                for item in res_data.get("classifications", []):
                    idx = item["index"]
                    if idx < len(batch):
                        q = batch[idx]
                        q["section"] = item["section"]
                        q["topic"] = item["topic"]
                        classified.append(q)
                break
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                retries -= 1
                time.sleep(2)
    
    print(f"Successfully classified {len(classified)} questions.")
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=2, ensure_ascii=False)
        
    print(f"Saved to {OUT_JSON.name}")

if __name__ == "__main__":
    main()
