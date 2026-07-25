# Mathematical proof of MAE scaling
sections = {
    "Verbal Ability": {"MAE": 1.734, "Topics": 11, "Total_Qs": 30},
    "Reasoning": {"MAE": 1.010, "Topics": 16, "Total_Qs": 25},
    "General Awareness": {"MAE": 0.400, "Topics": 46, "Total_Qs": 25},
    "Numerical Ability": {"MAE": 0.280, "Topics": 46, "Total_Qs": 20},
}

print("=== Relative Error Analysis ===")
for sec, data in sections.items():
    avg_qs = data["Total_Qs"] / data["Topics"]
    relative_error = data["MAE"] / avg_qs
    print(f"{sec:20} Avg Qs/Topic: {avg_qs:.2f} | MAE: {data['MAE']:.3f} | Relative Error: {relative_error*100:.1f}%")
