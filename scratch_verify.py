import cv2
import os

base = r'e:\afcatpyq3\data\images\AFCAT_2014_Official_Paper2'

for qnum in [66, 67, 68, 69, 70]:
    fname = f'venn_q{qnum}.png'
    p = os.path.join(base, fname)
    img = cv2.imread(p)
    if img is not None:
        print(f"Q{qnum}: {img.shape[1]}x{img.shape[0]} px")
    else:
        print(f"Q{qnum}: MISSING")

# Check Q66 and Q67 are different
img66 = cv2.imread(os.path.join(base, 'venn_q66.png'))
img67 = cv2.imread(os.path.join(base, 'venn_q67.png'))
if img66 is not None and img67 is not None:
    if img66.shape == img67.shape:
        diff = (img66 != img67).sum()
        print(f"\nQ66 vs Q67 pixel diff: {diff} (0=identical)")
    else:
        print(f"\nQ66 vs Q67: Different shapes = DIFFERENT images (GOOD!)")
