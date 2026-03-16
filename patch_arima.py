import codecs
import json

with codecs.open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# find "### ARIMA Model Training"
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and 'ARIMA Model Training' in cell['source'][0]:
        insert_idx = i + 1
        break

if insert_idx != -1:
    new_cell = {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"Training ARIMA Model...\")\n",
        "# order=(5,1,0) is a reasonable starting point for daily data that has been differenced\n",
        "arima_model = ARIMA(train_sales, order=(5, 1, 0))\n",
        "arima_model_fit = arima_model.fit()\n",
        "print(\"ARIMA training complete.\")\n"
      ]
    }
    nb['cells'].insert(insert_idx, new_cell)
    with codecs.open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f)

