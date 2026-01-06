#!/usr/bin/env python3
"""
Orchestration script: rebuild dashboard `data.js` from existing AI JSON outputs.

This script runs `scripts/mark_certain_topics.py` which computes `certainTopics`
from `ai_predicted_questions.json` and updates `output/predictions_2026/data.js`.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
DATA_JS = ROOT / 'output' / 'predictions_2026' / 'data.js'


def main():
    if not DATA_JS.exists():
        print('Warning: data.js not found at', DATA_JS)

    mark_script = SCRIPTS / 'mark_certain_topics.py'
    if not mark_script.exists():
        print('Error: mark_certain_topics.py not found at', mark_script)
        sys.exit(1)

    print('Running mark_certain_topics.py to compute certainTopics...')
    res = subprocess.run([sys.executable, str(mark_script)])
    if res.returncode != 0:
        print('mark_certain_topics.py failed with code', res.returncode)
        sys.exit(res.returncode)

    print('✅ data.js updated (certainTopics injected if present)')
    print('Location:', DATA_JS)


if __name__ == '__main__':
    main()
