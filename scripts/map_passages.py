import json
import fitz
import re
from pathlib import Path

def main():
    q_json_path = Path('data/processed/Q.json')
    papers_dir = Path('data/papers')
    
    # Load Q.json
    with open(q_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Group RC questions by file_name
    rc_questions_by_file = {}
    for i, q in enumerate(data):
        if q.get('topic') == 'Reading Comprehension' or q.get('section') == 'Verbal Ability':
            file_name = q.get('file_name')
            if file_name:
                if file_name not in rc_questions_by_file:
                    rc_questions_by_file[file_name] = []
                rc_questions_by_file[file_name].append((i, q))
                
    mapped_count = 0
    
    # Process each PDF
    for file_name, qs in rc_questions_by_file.items():
        pdf_path = papers_dir / file_name
        if not pdf_path.exists():
            continue
            
        try:
            doc = fitz.open(pdf_path)
            text = '\n'.join([p.get_text('text') for p in doc])
            doc.close()
            
            # Find passages using regex
            matches = re.finditer(r'(?i)direction.*?(?:qs?\.?\s*)?\(?\s*(\d+)\s*(?:-|to|and|\xad|\u2013|\u2014)\s*(\d+)\)?.*?passage.*?\n', text)
            for m in matches:
                start_q = int(m.group(1))
                end_q = int(m.group(2))
                start_idx = m.end()
                
                # Find where the actual question starts (e.g. '1.', 'Q1', '1 ')
                q_pattern = re.compile(rf'(?m)^\s*(?:Q\.?)?\s*{start_q}\s*[\.\)]')
                q_match = q_pattern.search(text[start_idx:])
                
                if q_match:
                    passage = text[start_idx:start_idx+q_match.start()].strip()
                    
                    # Map to questions in Q.json
                    for i, q in qs:
                        q_num = q.get('question_number')
                        if q_num and start_q <= q_num <= end_q:
                            data[i]['passage'] = passage
                            mapped_count += 1
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    # Save updated Q.json
    with open(q_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully mapped {mapped_count} passages to Reading Comprehension questions.")

if __name__ == '__main__':
    main()
