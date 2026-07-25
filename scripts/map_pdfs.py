import fitz
from pathlib import Path
import json

def extract_metadata(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text("text") + "\n"
        doc.close()
        
        # Clean text
        text = " ".join(text.split())
        
        # Heuristics for Sections
        sections = []
        if "Verbal" in text or "English" in text or "Synonym" in text: sections.append("VA")
        if "General Awareness" in text or "GK" in text: sections.append("GA")
        if "Reasoning" in text or "Spatial" in text: sections.append("RE")
        if "Numerical" in text or "Math" in text: sections.append("NA")
        
        # Heuristics for Shift
        shift = "Unknown"
        if "Shift 1" in text or "Shift I" in text or "Shift-1" in text or "Morning" in text: shift = "Shift 1"
        elif "Shift 2" in text or "Shift II" in text or "Shift-2" in text or "Afternoon" in text or "Evening" in text: shift = "Shift 2"
        
        # Date heuristics
        import re
        date_match = re.search(r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})', text, re.IGNORECASE)
        date = date_match.group(1) if date_match else "Unknown"
        
        return {
            "name": pdf_path.name,
            "text_snippet": text[:200],
            "detected_shift": shift,
            "detected_date": date,
            "detected_sections": sections
        }
    except Exception as e:
        return {"name": pdf_path.name, "error": str(e)}

def main():
    papers_dir = Path("e:/afcatpyq3/data/papers")
    pdfs = list(papers_dir.glob("*.pdf"))
    
    results = []
    for pdf in pdfs:
        res = extract_metadata(pdf)
        results.append(res)
        
    for r in results:
        print(f"[{r['name']}]")
        if "error" in r:
            print(f"  Error: {r['error']}")
        else:
            print(f"  Date: {r['detected_date']} | Shift: {r['detected_shift']} | Sections: {','.join(r['detected_sections'])}")
            
if __name__ == "__main__":
    main()
