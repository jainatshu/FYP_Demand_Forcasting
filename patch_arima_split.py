import json
with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r') as f:
    nb = json.load(f)

# Find ARIMA Model Training cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and len(cell.get('source', [])) > 0 and 'ARIMA Model Training' in getattr(cell, 'source', [''])[0]:
        pass # wait, cell['source'][0] might not be accurate.

# Better simply insert the split right before the ARIMA training code.
for i, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'arima_model' in source_text and 'ARIMA(' in source_text:
        new_source = [
            "print(\"Splitting daily_sales into train and test...\")\n",
            "train_size = int(len(daily_sales) * 0.8)\n",
            "train_sales, test_sales = daily_sales[:train_size], daily_sales[train_size:]\n",
            "\n"
        ]
        cell['source'] = new_source + cell['source']
        break

with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'w') as f:
    json.dump(nb, f)
