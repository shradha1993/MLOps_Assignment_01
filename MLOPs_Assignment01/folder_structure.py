from pathlib import Path

root = Path(__file__).resolve().parent

folders = [
    "notebook",
    "data/raw",
    "data/processed",
    "src/data",
    "src/features",
    "src/models",
    "src/tracking",
    "src/inference",
    "deployment",
    "test",
    "outputs/predictions",
    "outputs/reports",
]

for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

print(f"Created project structure at: {root}")
for folder in folders:
    print(f"  ✓ {folder}/")