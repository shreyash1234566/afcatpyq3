import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_questions_groq import GroqClient

def test():
    groq = GroqClient()
    print("Testing with keys:", len(groq.keys))
    print("Keys found:", groq.keys)
    print("Attempting to call Groq API...")
    
    resp = groq.chat(
        "You are an assistant. Respond ONLY in valid JSON.", 
        "Generate 1 simple question about dogs. Format: {\"questions\": [{\"question_text\": \"...\"}]}"
    )
    
    print("Response object:")
    print(resp)

if __name__ == "__main__":
    test()
