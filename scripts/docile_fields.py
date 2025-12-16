from docile.dataset import Dataset
d = Dataset('val', 'C:/Users/john_/code/docile/data/docile')
doc = d[0]
print(f'Doc: {doc.docid}')
print(f'Pages: {doc.page_count}')
print()
for f in doc.annotation.fields[:10]:
    print(f'{f.fieldtype}: \"{f.text}\" (page {f.page})')