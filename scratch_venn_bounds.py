import fitz

def get_venn_crop_bounds(pdf_path, q_start, q_next, q_prev=None):
    doc = fitz.open(pdf_path)
    print(f"\n--- {pdf_path.split('/')[-1]} ---")
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        
        q_start_y = None
        q_next_y = None
        q_prev_y = None
        
        # Find Y coords
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text.startswith(str(q_start) + ".") or text.startswith("Q" + str(q_start)):
                        q_start_y = line["bbox"][1]
                    if text.startswith(str(q_next) + ".") or text.startswith("Q" + str(q_next)):
                        q_next_y = line["bbox"][1]
                    if q_prev and (text.startswith(str(q_prev) + ".") or text.startswith("Q" + str(q_prev))):
                        q_prev_y = line["bbox"][1]
        
        if q_start_y is not None:
            print(f"Found Q{q_start} on page {page_idx+1} at Y={q_start_y}")
            if q_next_y:
                print(f"Found Q{q_next} on page {page_idx+1} at Y={q_next_y}")
            if q_prev_y:
                print(f"Found Q{q_prev} on page {page_idx+1} at Y={q_prev_y}")
            
            # Find (a), (b), (c), (d)
            options = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0: continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip() in ["(a)", "(b)", "(c)", "(d)"]:
                            options.append(line["bbox"][1])
            
            if options:
                min_opt = min(options)
                max_opt = max(options)
                print(f"Options found between Y={min_opt} and Y={max_opt}")
                
                # Determine where the options are
                if q_next_y and q_start_y < min_opt < q_next_y:
                    print("=> Options are INLINE (between Q_start and Q_next)")
                elif q_prev_y and q_prev_y < min_opt < q_start_y:
                    print("=> Options are SHARED (between Q_prev and Q_start)")
                elif min_opt < q_start_y:
                    print("=> Options are SHARED (above Q_start)")
                else:
                    print("=> Could not determine option placement")
            else:
                print("No options (a)(b)(c)(d) found on this page!")
            break

get_venn_crop_bounds(r'e:\afcatpyq3\data\papers\AFCAT_2014_Official_Paper2.pdf', 66, 67, 65)
get_venn_crop_bounds(r'e:\afcatpyq3\data\papers\AFCAT_2021_Memory.pdf', 71, 72, 70)
get_venn_crop_bounds(r'e:\afcatpyq3\data\papers\AFCAT_2015_Official_Paper2.pdf', 81, 82, 80)
