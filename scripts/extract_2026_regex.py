import fitz
import re
import json

pdf_path = r"e:\afcatpyq3\data\papers\AFCAT-I-Question-Paper-31-Jan-2026.pdf"
doc = fitz.open(pdf_path)

text = ""
for page in doc:
    text += page.get_text("text") + "\n"

# The text has "Q\n.\n1" or similar. Let's normalize it.
text = re.sub(r'Q\s*\.\s*(\d+)', r'Q.\1', text)
text = re.sub(r'A\s*n\s*s', 'Ans', text)

# Split by Q.\d+
chunks = re.split(r'(Q\.\d+)', text)

questions = []
current_q_num = None

for chunk in chunks:
    if re.match(r'Q\.\d+', chunk):
        current_q_num = chunk
    elif current_q_num and chunk.strip():
        # This is a question body
        q_text_match = re.search(r'(.*?)(?:Ans|Question ID)', chunk, flags=re.DOTALL)
        if q_text_match:
            q_text = q_text_match.group(1).strip()
            # Clean up newlines
            q_text = re.sub(r'\s+', ' ', q_text)
            
            # Find options
            opts_match = re.search(r'Ans(.*?)Question ID', chunk, flags=re.DOTALL)
            options = []
            if opts_match:
                opts_text = opts_match.group(1)
                opts = re.findall(r'\d\.\s*(.*?)(?=\d\.\s*|$)', opts_text, flags=re.DOTALL)
                options = [o.strip() for o in opts if o.strip()]
            
            questions.append({
                "q_num": current_q_num,
                "question_text": q_text,
                "options": options
            })

print(f"Extracted {len(questions)} questions.")
for q in questions[:3]:
    print(q['q_num'], q['question_text'][:100])

with open(r"e:\afcatpyq3\data\processed\Q_2026_extracted_raw.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)
