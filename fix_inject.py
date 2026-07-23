import re
from pathlib import Path

path = Path(r'e:\afcatpyq3\inject_images.py')
code = path.read_text(encoding='utf-8')

start_marker = '# 7. Patch index.html'
end_marker = 'print("\\n  [DONE]'

start_idx = code.find(start_marker)
end_idx = code.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    new_code = code[:start_idx] + code[end_idx:]
    path.write_text(new_code, encoding='utf-8')
    print("inject_images.py successfully patched!")
else:
    print("Error: Could not find markers in inject_images.py")
