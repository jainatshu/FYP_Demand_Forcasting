import json
with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print("--- CELL SOURCE ---")
        print("".join(cell['source']))
        if 'outputs' in cell and len(cell['outputs']) > 0:
            print("--- CELL OUTPUTS ---")
            for out in cell['outputs']:
                if 'text' in out:
                    print("".join(out['text']))
                elif 'traceback' in out:
                    print("".join(out['traceback']))
        break # just first code cell
