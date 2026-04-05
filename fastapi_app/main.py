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
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import json
from typing import List

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
        model = joblib.load(os.path.join(PARENT_DIR, 'rf_model_compressed.pkl.gz'))
    except Exception as e:
        print(f"Model load error: {e}")

    try:
        dtypes = {
            'Store': 'uint16', 'DayOfWeek': 'uint8', 'Sales': 'float32', 
            'Customers': 'float32', 'Open': 'uint8', 'Promo': 'uint8', 
            'StateHoliday': 'str', 'SchoolHoliday': 'uint8'
        }
        train_df = pd.read_csv(os.path.join(PARENT_DIR, 'datasets', 'train.csv'), parse_dates=['Date'], dtype=dtypes)
        store_df = pd.read_csv(os.path.join(PARENT_DIR, 'datasets', 'store.csv'))
        df_merged = pd.merge(train_df, store_df, on='Store', how='left')
        del train_df
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

class FilterRequest(BaseModel):
    store_type: List[str]
    assortment: List[str]
    school_holiday: str
    months: List[int]
    days: List[str]
    exclude_zero_sales: bool

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

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

    # 1. Sales Trends
    average_sales_time = df.groupby('DateStr')['Sales'].mean().reset_index()
    pct_change_sales_time = df.groupby('DateStr')['Sales'].sum().pct_change() * 100
    average_sales_time['Pct_Change'] = pct_change_sales_time.values
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=average_sales_time['DateStr'], y=average_sales_time['Sales'], name='Avg Sales', mode='lines+markers', line=dict(color='blue')), secondary_y=False)
    fig1.add_trace(go.Bar(x=average_sales_time['DateStr'], y=average_sales_time['Pct_Change'], name='% Change (MoM)', marker_color='rgba(0, 255, 0, 0.4)'), secondary_y=True)
    fig1.update_layout(title_text='Average Sales and Month-over-Month Percent Change')

    # 2. Day of Week
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_stats = df.groupby('DayName')['Sales'].mean().reindex(day_order).reset_index()
    fig2 = px.bar(day_stats, x='DayName', y='Sales', title='Average Sales by Day of Week', color='Sales', color_continuous_scale='viridis')
    fig2.add_hline(y=overall_mean_sales, line_dash='dash', line_color='red')

    # 3. Promo
    promo_stats = df.groupby('Promo')['Sales'].mean().reset_index()
    promo_stats['Promo_Label'] = promo_stats['Promo'].map({0: 'No Promo', 1: 'Promo Active'})
    fig3 = px.bar(promo_stats, x='Promo_Label', y='Sales', title='Promo Effectiveness', color='Promo_Label', color_discrete_sequence=['#3498db', '#e74c3c'], text=promo_stats['Sales'].apply(lambda x: f"${x:,.0f}"))

    # 4. Store
    store_stats = df.groupby('StoreType')['Sales'].mean().reset_index()
    fig4 = px.bar(store_stats, x='StoreType', y='Sales', title='Average Sales by Store Type', color='Sales', color_continuous_scale='magma')
    
    # 5. Assortment
    assort_stats = df.groupby('Assortment')['Sales'].mean().reset_index()
    fig5 = px.bar(assort_stats, x='Assortment', y='Sales', title='Average Sales by Assortment Type', color='Sales', color_continuous_scale='plasma')

    # 6. Holiday State
    hol_stats = df.groupby('StateHoliday')['Sales'].mean().reset_index()
    hol_stats['StateHoliday'] = hol_stats['StateHoliday'].astype(str)
    fig6 = px.bar(hol_stats, x='StateHoliday', y='Sales', title='Impact of State Holidays', color='StateHoliday', color_discrete_sequence=px.colors.sequential.Reds_r)

    # 7. Holiday School
    sh_stats = df.groupby('SchoolHoliday')['Sales'].mean().reset_index()
    sh_stats['SchoolHoliday'] = sh_stats['SchoolHoliday'].map({0: 'No', 1: 'Yes'})
    fig7 = px.bar(sh_stats, x='SchoolHoliday', y='Sales', title='Impact of School Holidays', color='SchoolHoliday', color_discrete_sequence=['#2ecc71', '#27ae60'])

    # 8. Combo
    combo_stats = df.groupby(['SchoolHoliday', 'Promo'])['Sales'].mean().reset_index()
    combo_stats['Condition'] = combo_stats.apply(lambda row: f"Promo: {'Yes' if row['Promo']==1 else 'No'} | School Hol: {'Yes' if row['SchoolHoliday']==1 else 'No'}", axis=1)
    fig8 = px.bar(combo_stats, x='Condition', y='Sales', color='Promo', title='Combined Effect of Holiday & Promo', text=combo_stats['Sales'].apply(lambda x: f"${x:,.0f}"), color_continuous_scale='Reds')

    # 9. Monthly School Holiday
    monthly_sh = df[df['SchoolHoliday'] == 1].groupby(['Month', 'Month_Name'])['Sales'].agg(total_holidays='count', avg_sales='mean').reset_index().sort_values('Month')
    fig9 = make_subplots(specs=[[{"secondary_y": True}]])
    if not monthly_sh.empty:
        fig9.add_trace(go.Bar(x=monthly_sh['Month_Name'], y=monthly_sh['total_holidays'], name='Holiday Days', marker_color='lightblue'), secondary_y=False)
        fig9.add_trace(go.Scatter(x=monthly_sh['Month_Name'], y=monthly_sh['avg_sales'], name='Avg Sales (During Holiday)', mode='lines+markers', line=dict(color='red')), secondary_y=True)
        fig9.update_layout(title_text='Month-wise School Holidays Effect')

    # 10. Monthly Promo
    monthly_promo = df[df['Promo'] == 1].groupby(['Month', 'Month_Name'])['Sales'].agg(total_promos='count', avg_sales='mean').reset_index().sort_values('Month')
    fig10 = make_subplots(specs=[[{"secondary_y": True}]])
    if not monthly_promo.empty:
        fig10.add_trace(go.Bar(x=monthly_promo['Month_Name'], y=monthly_promo['total_promos'], name='Promo Days', marker_color='orange'), secondary_y=False)
        fig10.add_trace(go.Scatter(x=monthly_promo['Month_Name'], y=monthly_promo['avg_sales'], name='Avg Sales (During Promo)', mode='lines+markers', line=dict(color='darkgreen')), secondary_y=True)
        fig10.update_layout(title_text='Month-wise Promotion Pattern')

    # 11. Corr Matrix
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Store', 'DayOfWeek', 'Year', 'Month', 'Promo2SinceYear', 'Promo2SinceWeek'], errors='ignore')
    corr_matrix = numeric_df.corr().round(2)
    fig11 = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Matrix of Numerical Features")

    charts = {
        "salesTrend": json.loads(pio.to_json(fig1)),
        "dayOfWeek": json.loads(pio.to_json(fig2)),
        "promo": json.loads(pio.to_json(fig3)),
        "store": json.loads(pio.to_json(fig4)),
        "assort": json.loads(pio.to_json(fig5)),
        "stateHol": json.loads(pio.to_json(fig6)),
        "schoolHol": json.loads(pio.to_json(fig7)),
        "combo": json.loads(pio.to_json(fig8)),
        "monthlySh": json.loads(pio.to_json(fig9)),
        "monthlyPromo": json.loads(pio.to_json(fig10)),
        "corr": json.loads(pio.to_json(fig11))
    }
    return charts

@app.post("/api/insights/filtered")
async def get_filtered_insights(req: FilterRequest):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    filtered_df = df.copy()
    if req.store_type:
        filtered_df = filtered_df[filtered_df['StoreType'].isin(req.store_type)]
    if req.assortment:
        filtered_df = filtered_df[filtered_df['Assortment'].isin(req.assortment)]
    if req.months:
        filtered_df = filtered_df[filtered_df['Month'].isin(req.months)]
    if req.days:
        filtered_df = filtered_df[filtered_df['DayName'].isin(req.days)]
        
    if req.school_holiday == "Yes":
        filtered_df = filtered_df[filtered_df['SchoolHoliday'] == 1]
    elif req.school_holiday == "No":
        filtered_df = filtered_df[filtered_df['SchoolHoliday'] == 0]
        
    if req.exclude_zero_sales:
        filtered_df = filtered_df[filtered_df['Sales'] > 0]

    record_count = len(filtered_df)
    if record_count == 0:
        return {
            "metrics": {"records": 0, "stores": 0, "avg_sales": 0},
            "charts": None
        }

    stores = int(filtered_df['Store'].nunique())
    avg_sales = float(filtered_df['Sales'].mean())

    fig_dist = px.histogram(filtered_df, x='Sales', nbins=50, title='Distribution of Sales', color_discrete_sequence=['#3498db'])
    
    promo_filtered = filtered_df.groupby('Promo')['Sales'].mean().reset_index()
    promo_filtered['Promo_Label'] = promo_filtered['Promo'].map({0: 'No Promo', 1: 'Promo Active'})
    fig_promo2 = px.bar(promo_filtered, x='Promo_Label', y='Sales', color='Promo_Label', title='Average Sales (Promo vs No Promo)', text=promo_filtered['Sales'].apply(lambda x: f"${x:,.0f}"))

    return {
        "metrics": {"records": record_count, "stores": stores, "avg_sales": avg_sales},
        "charts": {
            "dist": json.loads(pio.to_json(fig_dist)),
            "promo2": json.loads(pio.to_json(fig_promo2))
        }
    }

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
