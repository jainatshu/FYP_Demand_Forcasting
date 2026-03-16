import json
with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][8:18]):
    if cell['cell_type'] == 'code':
        print(f"--- Cell {i+8} ---")
        print("".join(cell.get('source', [])))
        print("-" * 40)
