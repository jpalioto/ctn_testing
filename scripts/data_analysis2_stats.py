import json
import statistics
from pathlib import Path

run_dir = Path("domains/extraction/results/run_20251217_053147/raw/claude-sonnet-4-5")
ctn_dir = run_dir / "ctn"
idio_dir = run_dir / "idiomatic"

deltas = []
ctn_scores = []
idio_scores = []

for ctn_file in ctn_dir.glob("*.json"):
    idio_file = idio_dir / ctn_file.name
    if idio_file.exists():
        ctn = json.loads(ctn_file.read_text())
        idio = json.loads(idio_file.read_text())
        ctn_scores.append(ctn["composite_score"])
        idio_scores.append(idio["composite_score"])
        deltas.append(ctn["composite_score"] - idio["composite_score"])

print(f"CTN:      mean={statistics.mean(ctn_scores):.3f}, std={statistics.stdev(ctn_scores):.3f}")
print(f"Idio:     mean={statistics.mean(idio_scores):.3f}, std={statistics.stdev(idio_scores):.3f}")
print(f"Delta:    mean={statistics.mean(deltas):.3f}, std={statistics.stdev(deltas):.3f}")
print("")
print(f"Delta range: {min(deltas):.3f} to {max(deltas):.3f}")
print(f"CTN wins: {sum(1 for d in deltas if d > 0)} / {len(deltas)}")
print(f"Ties:     {sum(1 for d in deltas if d == 0)} / {len(deltas)}")
print(f"Idio wins: {sum(1 for d in deltas if d < 0)} / {len(deltas)}")
