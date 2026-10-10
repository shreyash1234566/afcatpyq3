"""
extract_august_2026.py
======================
Extracts all 100 questions per shift (200 questions total) from the official
DigiALM AFCAT 02/2026 (August 2026) candidate response sheets:
  - AFCAT_2026_Aug_Shift1.pdf
  - AFCAT_2026_Aug_Shift2.pdf

Extracts:
  - q_num (1..100)
  - section (VA: 1-30, GA: 31-55, NA: 56-75, Reasoning: 76-100)
  - question_text
  - options (A, B, C, D)
  - correct_answer (A, B, C, D) derived from the DigiALM green tick mark
  - question_id, option_ids, status, chosen_option
  - figures / diagrams saved to data/images/2026_aug_s1/ and data/images/2026_aug_s2/
"""

import sys
import re
import json
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent.parent


def get_section(q_num: int) -> str:
    if 1 <= q_num <= 30:
        return "Verbal Ability"
    elif 31 <= q_num <= 55:
        return "General Awareness"
    elif 56 <= q_num <= 75:
        return "Numerical Ability"
    elif 76 <= q_num <= 100:
        return "Reasoning"
    return "Unknown"


def parse_digialm_pdf(pdf_path: Path, shift_id: str, img_dir: Path):
    doc = fitz.open(pdf_path)
    questions = []

    # First pass: map pages and collect text blocks and images
    # DigiALM structure: Each question starts with "Q.<num>"
    # followed by question text, then "Ans\n1. ...\n2. ...\n3. ...\n4. ..."
    # and a metadata table "Question ID : ...", "Status : ...", "Chosen Option : ..."

    # Pre-extract all text and images per page
    pages_data = []
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("blocks")

        # Find tick marks (w=16, h=16 PNGs)
        ticks = []
        large_images = []
        for img_info in page.get_images():
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            w, h = base_img["width"], base_img["height"]
            rects = page.get_image_rects(xref)
            if w == 16 and h == 16:
                ticks.extend(rects)
            elif w > 50 and h > 50:
                for r in rects:
                    large_images.append({
                        "xref": xref,
                        "w": w,
                        "h": h,
                        "rect": r,
                        "ext": base_img["ext"],
                        "bytes": base_img["image"]
                    })
        pages_data.append({
            "pno": pno,
            "blocks": blocks,
            "ticks": ticks,
            "large_images": large_images,
            "full_text": page.get_text()
        })

    # Find question locations across pages
    q_locs = {} # q_num -> list of {pno, block_idx, bbox, text}
    for pno, pdata in enumerate(pages_data):
        for bidx, b in enumerate(pdata["blocks"]):
            txt = b[4].strip()
            m = re.match(r"^Q\.(\d+)", txt)
            if m:
                q_num = int(m.group(1))
                q_locs[q_num] = {"pno": pno, "bidx": bidx, "bbox": fitz.Rect(b[:4])}

    print(f"[{shift_id}] Found {len(q_locs)} questions (Q.1 to Q.{max(q_locs.keys()) if q_locs else 0})")

    # Now parse each question from its start to the next question start
    sorted_q_nums = sorted(q_locs.keys())

    for idx, q_num in enumerate(sorted_q_nums):
        start_info = q_locs[q_num]
        start_pno = start_info["pno"]
        start_y = start_info["bbox"].y0

        if idx + 1 < len(sorted_q_nums):
            next_q = sorted_q_nums[idx + 1]
            end_info = q_locs[next_q]
            end_pno = end_info["pno"]
            end_y = end_info["bbox"].y0
        else:
            end_pno = len(doc) - 1
            end_y = 999999.0

        # Collect blocks belonging to this question across start_pno .. end_pno
        q_blocks = []
        q_ticks = []
        q_images = []

        for pno in range(start_pno, end_pno + 1):
            pdata = pages_data[pno]
            for b in pdata["blocks"]:
                b_rect = fitz.Rect(b[:4])
                # Filter by vertical bounds
                if pno == start_pno and b_rect.y1 < start_y:
                    continue
                if pno == end_pno and b_rect.y0 >= end_y:
                    continue
                # Exclude header/footer info
                if b[4].strip().startswith("AFCAT 02") or "Hall Ticket No" in b[4] or "Application Number" in b[4] or "Section :" in b[4]:
                    continue
                q_blocks.append((pno, b_rect, b[4]))

            for t_rect in pdata["ticks"]:
                if pno == start_pno and t_rect.y1 < start_y:
                    continue
                if pno == end_pno and t_rect.y0 >= end_y:
                    continue
                q_ticks.append((pno, t_rect))

            for img_dict in pdata["large_images"]:
                r = img_dict["rect"]
                if pno == start_pno and r.y1 < start_y:
                    continue
                if pno == end_pno and r.y0 >= end_y:
                    continue
                q_images.append((pno, img_dict))

        # Separate question stem, options, and metadata
        q_text_lines = []
        opt_lines = {1: [], 2: [], 3: [], 4: []}
        meta_lines = []
        opt_rects = {} # opt_num -> list of Rects

        current_opt = None
        in_ans = False

        for pno, b_rect, txt in q_blocks:
            lines = [l.strip() for l in txt.split("\n") if l.strip()]
            for line in lines:
                if re.match(r"^Q\.\d+", line):
                    # Strip leading Q.N
                    rem = re.sub(r"^Q\.\d+\s*", "", line).strip()
                    if rem: q_text_lines.append(rem)
                    continue

                if line == "Ans":
                    in_ans = True
                    continue

                # Check option indicators
                m_opt = re.match(r"^([1-4])\.\s*(.*)", line)
                if in_ans and m_opt:
                    current_opt = int(m_opt.group(1))
                    opt_text = m_opt.group(2).strip()
                    if opt_text:
                        opt_lines[current_opt].append(opt_text)
                    if current_opt not in opt_rects:
                        opt_rects[current_opt] = (pno, b_rect)
                    continue

                # Check metadata
                if any(k in line for k in ["Question ID :", "Option 1 ID :", "Option 2 ID :",
                                           "Option 3 ID :", "Option 4 ID :", "Status :", "Chosen Option :"]):
                    meta_lines.append(line)
                    continue

                if current_opt is not None and in_ans:
                    opt_lines[current_opt].append(line)
                elif not in_ans:
                    q_text_lines.append(line)

        # Build full question stem text
        full_q_text = " ".join(q_text_lines).strip()
        # Clean extra spaces
        full_q_text = re.sub(r"\s+", " ", full_q_text)

        # Parse metadata
        meta_blob = "\n".join(meta_lines)
        qid_m = re.search(r"Question ID\s*:\s*(\d+)", meta_blob)
        status_m = re.search(r"Status\s*:\s*([^\n]+)", meta_blob)
        chosen_m = re.search(r"Chosen Option\s*:\s*([^\n]+)", meta_blob)

        question_id = qid_m.group(1) if qid_m else None
        status = status_m.group(1).strip() if status_m else "Not Answered"
        chosen_option = chosen_m.group(1).strip() if chosen_m else "--"

        # Determine option text or image references
        options = {}
        for num, key in [(1, "A"), (2, "B"), (3, "C"), (4, "D")]:
            val = " ".join(opt_lines[num]).strip()
            val = re.sub(r"\s+", " ", val)
            options[key] = val

        # Handle diagrams and figures
        has_figure = len(q_images) > 0
        fig_paths = []
        for i_idx, (pno, img_dict) in enumerate(q_images):
            # Save figure image
            img_filename = f"q{q_num}_fig_{i_idx+1}.{img_dict['ext']}"
            img_out = img_dir / img_filename
            if not img_out.exists():
                img_out.write_bytes(img_dict["bytes"])
            rel_path = f"data/images/{img_dir.name}/{img_filename}"
            fig_paths.append(rel_path)

        # If options are empty but images exist, link images to options
        if not options["A"] and len(q_images) >= 4:
            # The images are the 4 options!
            sorted_by_y = sorted(q_images, key=lambda x: (x[0], x[1]["rect"].y0))
            for k_idx, key in enumerate(["A", "B", "C", "D"]):
                if k_idx < len(sorted_by_y):
                    img_dict = sorted_by_y[k_idx][1]
                    img_filename = f"q{q_num}_opt_{key}.{img_dict['ext']}"
                    img_out = img_dir / img_filename
                    if not img_out.exists():
                        img_out.write_bytes(img_dict["bytes"])
                    options[key] = f"[Figure: data/images/{img_dir.name}/{img_filename}]"
            # If there was a 5th image, it was the question figure
            if len(sorted_by_y) > 4:
                q_fig_dict = sorted_by_y[0][1]
                q_fig_out = img_dir / f"q{q_num}_stem.{q_fig_dict['ext']}"
                if not q_fig_out.exists():
                    q_fig_out.write_bytes(q_fig_dict["bytes"])
                fig_paths = [f"data/images/{img_dir.name}/q{q_num}_stem.{q_fig_dict['ext']}"]

        # Determine correct answer from tick location
        correct_answer = "A" # fallback
        if q_ticks:
            # Match tick to closest option rectangle
            tick_pno, tick_rect = q_ticks[0]
            tick_mid_y = (tick_rect.y0 + tick_rect.y1) / 2.0

            best_opt = None
            min_dist = float("inf")

            for num in [1, 2, 3, 4]:
                if num in opt_rects:
                    o_pno, o_rect = opt_rects[num]
                    if o_pno == tick_pno:
                        dist = abs((o_rect.y0 + o_rect.y1) / 2.0 - tick_mid_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_opt = num

            if best_opt:
                correct_answer = {1: "A", 2: "B", 3: "C", 4: "D"}[best_opt]
            else:
                # Fallback: estimate from vertical quartiles
                all_y = sorted([r.y0 for p, r in opt_rects.values()]) if opt_rects else []
                if all_y:
                    closest_idx = min(range(len(all_y)), key=lambda i: abs(all_y[i] - tick_mid_y))
                    correct_answer = ["A", "B", "C", "D"][closest_idx]

        q_record = {
            "question_number": q_num,
            "section": get_section(q_num),
            "question_text": full_q_text,
            "options": options,
            "correct_answer": correct_answer,
            "question_id": question_id,
            "status": status,
            "chosen_option": chosen_option,
            "has_figure": has_figure,
            "image_path": fig_paths[0] if fig_paths else None,
            "all_images": fig_paths,
            "shift": shift_id
        }
        questions.append(q_record)

    return questions


def main():
    s1_pdf = ROOT / "data" / "papers" / "AFCAT_2026_Aug_Shift1.pdf"
    s2_pdf = ROOT / "data" / "papers" / "AFCAT_2026_Aug_Shift2.pdf"

    s1_img_dir = ROOT / "data" / "images" / "2026_aug_s1"
    s2_img_dir = ROOT / "data" / "images" / "2026_aug_s2"

    print("Parsing Shift 1...")
    s1_qs = parse_digialm_pdf(s1_pdf, "Shift 1", s1_img_dir)
    print(f"Shift 1 parsed: {len(s1_qs)} questions")

    out_s1 = ROOT / "data" / "papers" / "afcat_2026_aug_shift1_raw.json"
    with open(out_s1, "w", encoding="utf-8") as f:
        json.dump(s1_qs, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_s1}")

    print("\nParsing Shift 2...")
    s2_qs = parse_digialm_pdf(s2_pdf, "Shift 2", s2_img_dir)
    print(f"Shift 2 parsed: {len(s2_qs)} questions")

    out_s2 = ROOT / "data" / "papers" / "afcat_2026_aug_shift2_raw.json"
    with open(out_s2, "w", encoding="utf-8") as f:
        json.dump(s2_qs, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_s2}")


if __name__ == "__main__":
    main()
