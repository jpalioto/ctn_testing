from ctn_testing.core.loaders import load_docile

docs = load_docile("C:/Users/john_/code/docile/data/docile", split="val", n=3)
print(f"Loaded {len(docs)} documents")
for d in docs:
    print(f"  {d.id}: {len(d.ground_truth)} fields")
    for name, gt in list(d.ground_truth.items())[:3]:
        print(f"    {name}: {gt.value[:30] if gt.value else None}...")
