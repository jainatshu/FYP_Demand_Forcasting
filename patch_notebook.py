import json

notebook_path = "04_Model_Building_Evaluation.ipynb"

try:
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                # Replace imports
                if "import xgboost as xgb" in line:
                    new_source.append("# import xgboost as xgb\n")
                    new_source.append("from sklearn.ensemble import GradientBoostingRegressor\n")
                elif "from xgboost import XGBRegressor" in line:
                    new_source.append("# from xgboost import XGBRegressor\n")
                
                # Replace usage
                elif "XGBRegressor" in line:
                    # Replace XGBRegressor with GradientBoostingRegressor
                    line = line.replace("XGBRegressor", "GradientBoostingRegressor")
                    # Remove n_jobs=-1 as it's not a valid param for GradientBoostingRegressor
                    line = line.replace(", n_jobs=-1", "")
                    line = line.replace("n_jobs=-1", "")
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell["source"] = new_source

    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=4)
    
    print("Successfully patched notebook.")

except Exception as e:
    print(f"Error patching notebook: {e}")
