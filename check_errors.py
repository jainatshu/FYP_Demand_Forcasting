import json
with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r') as f:
    nb = json.load(f)
found_error = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                print(f'Error in cell {i}:')
                print(''.join(out.get('traceback', [])))
                found_error = True
if not found_error:
    print('SUCCESS: No errors found.')
