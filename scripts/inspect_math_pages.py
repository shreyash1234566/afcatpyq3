import fitz

doc = fitz.open('data/papers/AFCAT_2026_Aug_Shift1.pdf')

for pno in [20, 22, 23]:
    page = doc[pno]
    words = page.get_text("words")
    print(f"=== Page {pno} Words ({len(words)}) ===")
    lines = page.get_text("text").splitlines()
    for l in lines:
        if any(k in l for k in ["Q.65", "Q.70", "Q.72", "Q.74", "Q.75", "Ans", "Question ID"]):
            print("  ", l)
