from pathlib import Path

root = Path(r"C:\Users\shrad\MLOps_Assignment_01\MLOPs_Assignment01")

folders = [
    "notebook",
    "data/raw",
    "data/processed",
    "src",
    "models",
    "deployment",
    "airflow",
    "outputs/predictions",
    "outputs/reports",
]

for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

print(f"Created project structure at: {root}")
for folder in folders:
    print(f"  ✓ {folder}/")