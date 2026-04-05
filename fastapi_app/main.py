from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import json

app = FastAPI(title="Rossmann Store Interface")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

model = None
df = None
INVENTORY_FILE = os.path.join(PARENT_DIR, 'datasets', 'inventory.csv')

@app.on_event("startup")
def startup_event():
    global model, df
    try:
        model = joblib.load(os.path.join(PARENT_DIR, 'rf_model.pkl'))
    except Exception as e:
        print(f"Model load error: {e}")

    try:
        train_df = pd.read_csv(os.path.join(PARENT_DIR, 'datasets', 'train.csv'), parse_dates=['Date'])
        store_df = pd.read_csv(os.path.join(PARENT_DIR, 'datasets', 'store.csv'))
        df_merged = pd.merge(train_df, store_df, on='Store', how='left')
        df_merged['Year'] = df_merged['Date'].dt.year
        df_merged['Month'] = df_merged['Date'].dt.month
        df_merged['Month_Name'] = df_merged['Date'].dt.month_name()
        df_merged['DateStr'] = df_merged['Date'].dt.strftime('%Y-%m')
        df_merged['DayName'] = df_merged['Date'].dt.day_name().str[:3]
        df = df_merged
    except Exception as e:
        print(f"Data load error: {e}")

class PredictRequest(BaseModel):
    store_type: int
    assortment: int
    comp_dist: float
    comp_month: int
    comp_year: int
    promo: int
    promo2_year: int
    promo2_week: int
    school_holiday: int
    day_of_week: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict_sales(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    days = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0, "Friday": 0, "Saturday": 0}
    if req.day_of_week in days:
        days[req.day_of_week] = 1
        
    features = pd.DataFrame([{
        'CompetitionDistance': req.comp_dist,
        'Promo': req.promo,
        'CompetitionOpenSinceMonth': req.comp_month,
        'CompetitionOpenSinceYear': req.comp_year,
        'StoreType': req.store_type,
        'Promo2SinceYear': req.promo2_year,
        'Day_1': days["Monday"],
        'Day_2': days["Tuesday"],
        'Day_3': days["Wednesday"],
        'Day_4': days["Thursday"],
        'Day_5': days["Friday"],
        'Day_6': days["Saturday"],
        'Promo2SinceWeek': req.promo2_week,
        'Assortment': req.assortment,
        'SchoolHoliday': req.school_holiday
    }])
    
    pred = model.predict(features)[0]
    return {"prediction": float(pred)}

@app.get("/api/insights/overall")
async def get_overall_insights():
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    overall_mean_sales = df['Sales'].mean()

    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_stats = df.groupby('DayName')['Sales'].mean().reindex(day_order).reset_index()
    fig_day = px.bar(day_stats, x='DayName', y='Sales', title='Avg Sales by Day', color='Sales', color_continuous_scale='viridis')
    fig_day.add_hline(y=overall_mean_sales, line_dash='dash', line_color='red')

    promo_stats = df.groupby('Promo')['Sales'].mean().reset_index()
    promo_stats['Promo_Label'] = promo_stats['Promo'].map({0: 'No Promo', 1: 'Promo Active'})
    fig_promo = px.bar(promo_stats, x='Promo_Label', y='Sales', title='Promo Effectiveness', color='Promo_Label', color_discrete_sequence=['#3498db', '#e74c3c'], text=promo_stats['Sales'].apply(lambda x: f"${x:,.0f}"))

    store_stats = df.groupby('StoreType')['Sales'].mean().reset_index()
    fig_store = px.bar(store_stats, x='StoreType', y='Sales', title='Avg Sales by Store Type', color='Sales', color_continuous_scale='magma')
    
    assort_stats = df.groupby('Assortment')['Sales'].mean().reset_index()
    fig_assort = px.bar(assort_stats, x='Assortment', y='Sales', title='Avg Sales by Assortment Type', color='Sales', color_continuous_scale='plasma')

    charts = {
        "dayOfWeek": json.loads(pio.to_json(fig_day)),
        "promo": json.loads(pio.to_json(fig_promo)),
        "store": json.loads(pio.to_json(fig_store)),
        "assort": json.loads(pio.to_json(fig_assort))
    }
    return charts

@app.get("/api/inventory")
async def get_inventory():
    if not os.path.exists(INVENTORY_FILE):
        return []
    inv_df = pd.read_csv(INVENTORY_FILE)
    return inv_df.to_dict(orient='records')

class InventoryItem(BaseModel):
    Item_ID: int
    Item_Name: str
    Category: str
    Stock: int
    Reorder_Level: int
    Unit_Price: float

@app.post("/api/inventory")
async def modify_inventory(item: InventoryItem):
    if not os.path.exists(INVENTORY_FILE):
        inv_df = pd.DataFrame(columns=["Item_ID","Item_Name","Category","Stock","Reorder_Level","Unit_Price"])
    else:
        inv_df = pd.read_csv(INVENTORY_FILE)

    idx_match = inv_df.index[inv_df['Item_ID'] == item.Item_ID].tolist()
    if len(idx_match) > 0:
        inv_df.loc[idx_match[0], ["Item_ID", "Item_Name", "Category", "Stock", "Reorder_Level", "Unit_Price"]] = [item.Item_ID, item.Item_Name, item.Category, item.Stock, item.Reorder_Level, item.Unit_Price]
    else:
        new_row = pd.DataFrame([{ "Item_ID": item.Item_ID, "Item_Name": item.Item_Name, "Category": item.Category, "Stock": item.Stock, "Reorder_Level": item.Reorder_Level, "Unit_Price": item.Unit_Price }])
        inv_df = pd.concat([inv_df, new_row], ignore_index=True)
    
    inv_df.to_csv(INVENTORY_FILE, index=False)
    return {"message": "Success"}

@app.delete("/api/inventory/{item_id}")
async def delete_inventory(item_id: int):
    if os.path.exists(INVENTORY_FILE):
        inv_df = pd.read_csv(INVENTORY_FILE)
        inv_df = inv_df[inv_df['Item_ID'] != item_id]
        inv_df.to_csv(INVENTORY_FILE, index=False)
    return {"message": "Deleted"}
