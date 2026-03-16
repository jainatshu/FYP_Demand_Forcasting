import json
with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'r') as f:
    nb = json.load(f)

new_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "import sys\n",
    "!{sys.executable} -m pip install statsmodels\n"
  ]
}
nb['cells'].insert(0, new_cell)

with open('Time_Series_Analysis/Time_Series_Analysis.ipynb', 'w') as f:
    json.dump(nb, f)
