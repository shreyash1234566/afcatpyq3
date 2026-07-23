import re
from pathlib import Path

html_path = Path(r'e:\afcatpyq3\output\predictions_2026\index.html')
html = html_path.read_text(encoding='utf-8')

# We want to replace EVERYTHING between the question text paragraph and the options rendering block.
start_marker = '<p class="text-xl font-bold text-brand-navy mb-10 leading-relaxed font-heading">${q.question_text || q.question}</p>'
end_marker = '<div class="grid md:grid-cols-2 gap-4">'

start_idx = html.find(start_marker)
end_idx = html.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    clean_image_snippet = """
                ${(q.has_figure && q.image_path && q.image_path.length > 0) ? `
                <div style="margin:12px 0 24px 0;padding:12px;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;display:flex;flex-wrap:wrap;gap:10px;justify-content:center;align-items:center;">
                    ${q.image_path.map((imgSrc,ii) => \`<img src="${imgSrc}" alt="Figure ${ii+1}" loading="lazy" style="${q.image_dark ? 'filter:invert(1) brightness(0.85);background:#111;' : 'background:#fff;'} max-height:180px;max-width:100%;object-fit:contain;border-radius:8px;padding:4px;" onerror="this.style.display='none'" />\`).join('')}
                </div>` : ''}
                """
    
    new_html = html[:start_idx + len(start_marker)] + "\n" + clean_image_snippet + "\n                " + html[end_idx:]
    html_path.write_text(new_html, encoding='utf-8')
    print("index.html successfully cleaned and patched!")
else:
    print("Error: Could not find markers in index.html")
