import json, re, os

# Check temp_data.js
fsize = os.path.getsize(r'E:\afcatpyq3\temp_data.js')
print(f'temp_data.js size: {fsize/1024:.1f} KB')

html = open(r'E:\afcatpyq3\output\predictions_2026\index.html', encoding='utf-8').read()
scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', html)
print('Script sources:', scripts)

# Find what data sources the HTML uses for questions
# Look for pyqData, Q.json, temp_data, etc
for keyword in ['pyqData', 'Q.json', 'temp_data', 'question_bank', 'all_questions', 'mock_test']:
    count = html.count(keyword)
    if count:
        print(f'{keyword}: found {count} times')
        # show one context
        idx = html.find(keyword)
        print('  Context:', html[max(0,idx-50):idx+80].replace('\n',' '))
