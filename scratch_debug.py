import fitz
doc = fitz.open(r'e:\afcatpyq3\data\papers\AFCAT_2014_Official_Paper2.pdf')
page = doc[12]

q66_y = None
q67_y = None

text_dict = page.get_text("dict")
for block in text_dict.get("blocks", []):
    if block.get("type") != 0: continue
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        if not spans: continue
        line_text = "".join(s["text"] for s in spans).strip()
        if line_text.startswith("Q66"):
            q66_y = line["bbox"][1]
        elif line_text.startswith("Q67"):
            q67_y = line["bbox"][1]

print(f"Q66 Y: {q66_y}")
print(f"Q67 Y: {q67_y}")

if q66_y and q67_y:
    pix = page.get_pixmap(clip=fitz.Rect(0, q66_y, page.rect.width, q67_y))
    pix.save(r'e:\afcatpyq3\test_crop.png')
    print("Saved test_crop.png")
