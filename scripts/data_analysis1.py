import json
from pathlib import Path

run_dir = Path("domains/extraction/results/run_20251217_053147/raw/claude-sonnet-4-5")
ctn_dir = run_dir / "ctn"
idio_dir = run_dir / "idiomatic"

deltas = []
for ctn_file in ctn_dir.glob("*.json"):
    idio_file = idio_dir / ctn_file.name
    if idio_file.exists():
        ctn = json.loads(ctn_file.read_text())
        idio = json.loads(idio_file.read_text())
        delta = ctn["composite_score"] - idio["composite_score"]
        deltas.append((ctn_file.stem, ctn["composite_score"], idio["composite_score"], delta))

deltas.sort(key=lambda x: x[3], reverse=True)
print("TOP 10 CTN WINS:")
for doc, c, i, d in deltas[:10]:
    print(f"  {doc[:12]}: CTN {c:.2f} vs Idio {i:.2f} (Δ={d:+.2f})")

print()
print("TOP 10 IDIOMATIC WINS:")
for doc, c, i, d in deltas[-10:]:
    print(f"  {doc[:12]}: CTN {c:.2f} vs Idio {i:.2f} (Δ={d:+.2f})")
