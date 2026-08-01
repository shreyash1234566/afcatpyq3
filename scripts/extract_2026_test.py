import fitz
import sys

pdf_path = r"e:\afcatpyq3\data\papers\AFCAT-I-Question-Paper-31-Jan-2026.pdf"
try:
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    text = ""
    for i in range(min(3, len(doc))):
        text += f"\n--- Page {i+1} ---\n"
        text += doc[i].get_text("text")
    print(text[:2000])
except Exception as e:
    print("Error:", e)
