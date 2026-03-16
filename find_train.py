import json
with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'train' in source and 'test' in source and 'split' in source.lower():
            print(f"--- Cell {i} (Train/Test Split) ---")
            print(source)
            print("-" * 40)
        elif 'train' in source and 'data' in source:
            print(f"--- Cell {i} (Possible training data) ---")
            print(source)
            print("-" * 40)
