import fitz
from PIL import Image
import io

doc = fitz.open(r'e:\afcatpyq3\data\papers\AFCAT_2014_Official_Paper2.pdf')

# Page 13: crop from Q66 (400) to bottom
page13 = doc[12]
pix13 = page13.get_pixmap(clip=fitz.Rect(0, 400, page13.rect.width, page13.rect.height))
img13 = Image.open(io.BytesIO(pix13.tobytes()))

# Page 14: crop from top to Q67 (approx 120)
page14 = doc[13]
# Find Q67 y
q67_y = 120
for block in page14.get_text("dict").get("blocks", []):
    if block.get("type") == 0:
        for line in block.get("lines", []):
            if line.get("spans") and "Q67" in line["spans"][0]["text"]:
                q67_y = line["bbox"][1]

pix14 = page14.get_pixmap(clip=fitz.Rect(0, 0, page14.rect.width, q67_y))
img14 = Image.open(io.BytesIO(pix14.tobytes()))

# Stitch
dst = Image.new('RGB', (img13.width, img13.height + img14.height))
dst.paste(img13, (0, 0))
dst.paste(img14, (0, img13.height))
dst.save(r'e:\afcatpyq3\test_stitched.png')
print("Saved stitched image!")
