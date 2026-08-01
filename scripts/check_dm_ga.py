import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dirichlet_forecaster import DirichletForecaster
from scripts.eval_end_to_end_2024 import load_counts_up_to_2023

cnt, syt, years = load_counts_up_to_2023()
model = DirichletForecaster(cnt, syt, years)
plan = model.predict(sections=['General Awareness'])
for t in plan['General Awareness']['topics']:
    print(f"{t['topic']}: {t['expected_count_exact']}")
