import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import os

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Rossmann Sales Prediction", page_icon="📈", layout="wide")

# ---- STYLES ----
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; background-color: #ff4b4b; color: white; }
    h1, h2, h3 { color: #2c3e50; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }
    .metric-value { font-size: 36px; font-weight: bold; color: #ff4b4b; }
    .metric-label { font-size: 16px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ---- DATA LOADING ----
@st.cache_resource
def load_model():
    return joblib.load('rf_model_compressed.pkl.gz')

@st.cache_data
def load_and_prep_data():
    # Load raw data for insights
    dtypes = {
        'Store': 'uint16', 'DayOfWeek': 'uint8', 'Sales': 'float32', 
        'Customers': 'float32', 'Open': 'uint8', 'Promo': 'uint8', 
        'StateHoliday': 'str', 'SchoolHoliday': 'uint8'
    }
    train_df = pd.read_csv('datasets/train.csv', parse_dates=['Date'], dtype=dtypes)
    store_df = pd.read_csv('datasets/store.csv')
    
    # Merge datasets
    df = pd.merge(train_df, store_df, on='Store', how='left')
    del train_df
    
    # Feature Engineering for Insights
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.month_name()
    df['DateStr'] = df['Date'].dt.strftime('%Y-%m')
    df['DayName'] = df['Date'].dt.day_name().str[:3]
    
    return df

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = "Prediction"

# Load the resources
try:
    rf_model = load_model()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading model: {e}. Please ensure rf_model.pkl exists.")

try:
    df = load_and_prep_data()
    raw_data_loaded = True
except Exception as e:
    raw_data_loaded = False
    st.warning(f"Could not load raw datasets for insights. Error: {e}")

# ---- SIDEBAR NAVIGATION ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🔮 Sales Prediction", "📊 Overall Insights", "🔍 Filtered Insights", "📦 Inventory Management"])

# ==========================================================
# PAGE 1: PREDICTION
# ==========================================================
if page == "🔮 Sales Prediction":
    st.title("🔮 Rossmann Store Sales Prediction")
    st.markdown("Enter the store details below to predict the expected sales using the trained Random Forest model.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Store Info")
        store_type = st.selectbox("Store Type", options=[1, 2, 3, 4], help="1=a, 2=b, 3=c, 4=d (Based on processing)")
        assortment = st.selectbox("Assortment Level", options=[1, 2, 3], help="1=Basic, 2=Extra, 3=Extended")
        comp_dist = st.number_input("Competition Distance (meters)", min_value=0.0, value=1200.0, step=50.0)
        comp_month = st.number_input("Competition Open Since (Month)", min_value=1, max_value=12, value=1)
        comp_year = st.number_input("Competition Open Since (Year)", min_value=1900, max_value=2026, value=2010)
        
    with col2:
        st.subheader("Promotions & Holidays")
        promo = st.radio("Is Promo Active?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        promo2_year = st.number_input("Promo2 Since Year", min_value=1900, max_value=2026, value=2010)
        promo2_week = st.number_input("Promo2 Since Week", min_value=1, max_value=52, value=14)
        school_holiday = st.radio("Is School Holiday?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        
    with col3:
        st.subheader("Date & Time Info")
        month = st.selectbox("Select Month", options=list(range(1, 13)))
        day_of_week = st.selectbox("Select Day of the Week", 
                                   options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
        
    st.markdown("---")
    
    # Predict button
    if st.button("Predict Expected Sales 🚀"):
        if data_loaded:
            # Prepare Day Booleans
            days = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0, "Friday": 0, "Saturday": 0}
            days[day_of_week] = 1
            
            # Match top features used for the model
            features = pd.DataFrame([{
                'CompetitionDistance': comp_dist,
                'Promo': promo,
                'CompetitionOpenSinceMonth': comp_month,
                'CompetitionOpenSinceYear': comp_year,
                'StoreType': store_type,
                'Promo2SinceYear': promo2_year,
                'Day_1': days["Monday"],
                'Day_2': days["Tuesday"],
                'Day_3': days["Wednesday"],
                'Day_4': days["Thursday"],
                'Day_5': days["Friday"],
                'Day_6': days["Saturday"],
                'Promo2SinceWeek': promo2_week,
                'Assortment': assortment,
                'SchoolHoliday': school_holiday
            }])
            
            # Make prediction
            prediction = rf_model.predict(features)[0]
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Sales</div>
                <div class="metric-value">${prediction:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"Predicted successfully for the month of {pd.to_datetime(str(month), format='%m').month_name()}!")
            st.balloons()
        else:
            st.error("Model is not loaded. Cannot make predictions.")

# ==========================================================
# PAGE 2: OVERALL INSIGHTS
# ==========================================================
elif page == "📊 Overall Insights":
    st.title("📊 Actionable Insights Dashboard")
    st.markdown("Explore deep insights extracted from historical Rossmann Sales data.")
    
    if raw_data_loaded:
        overall_mean_sales = df['Sales'].mean()
        
        # 1. Sales Trends Over Time
        st.subheader("� 1. Sales Trends Over Time")
        st.markdown("Examine how average sales and total sales percentage change fluctuate on a month-by-month basis.")
        
        average_sales_time = df.groupby('DateStr')['Sales'].mean().reset_index()
        pct_change_sales_time = df.groupby('DateStr')['Sales'].sum().pct_change() * 100
        average_sales_time['Pct_Change'] = pct_change_sales_time.values
        
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=average_sales_time['DateStr'], y=average_sales_time['Sales'], name='Avg Sales', mode='lines+markers', line=dict(color='blue')), secondary_y=False)
        fig1.add_trace(go.Bar(x=average_sales_time['DateStr'], y=average_sales_time['Pct_Change'], name='% Change (MoM)', marker_color='rgba(0, 255, 0, 0.4)'), secondary_y=True)
        fig1.update_layout(title_text='Average Sales and Month-over-Month Percent Change')
        fig1.update_yaxes(title_text="Average Sales", secondary_y=False)
        fig1.update_yaxes(title_text="% Change", secondary_y=True)
        st.plotly_chart(fig1, width='stretch')
        
        col_a, col_b = st.columns(2)
        
        # 2. Day of Week Insights
        with col_a:
            st.subheader("📅 2. Day of Week Insights")
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_stats = df.groupby('DayName')['Sales'].mean().reindex(day_order).reset_index()
            
            fig2 = px.bar(day_stats, x='DayName', y='Sales', title='Average Sales by Day of Week', color='Sales', color_continuous_scale='viridis')
            fig2.add_hline(y=overall_mean_sales, line_dash='dash', line_color='red', annotation_text='Overall Avg')
            st.plotly_chart(fig2, width='stretch')
            
        # 3. Promotional Insights
        with col_b:
            st.subheader("🏷️ 3. Impact of Promotions")
            promo_stats = df.groupby('Promo')['Sales'].mean().reset_index()
            promo_stats['Promo_Label'] = promo_stats['Promo'].map({0: 'No Promo', 1: 'Promo Active'})
            
            promo_increase = (promo_stats[promo_stats['Promo']==1]['Sales'].values[0] - promo_stats[promo_stats['Promo']==0]['Sales'].values[0]) / promo_stats[promo_stats['Promo']==0]['Sales'].values[0] * 100
            
            fig3 = px.bar(promo_stats, x='Promo_Label', y='Sales', title=f'Sales Increase: {promo_increase:.2f}%', color='Promo_Label', color_discrete_sequence=['#3498db', '#e74c3c'], text=promo_stats['Sales'].apply(lambda x: f"${x:,.0f}"))
            st.plotly_chart(fig3, width='stretch')

        col_c, col_d = st.columns(2)
        
        # 4. Store Type Performance
        with col_c:
            st.subheader("🏪 4. Store Type Performance")
            store_stats = df.groupby('StoreType')['Sales'].mean().reset_index()
            fig4 = px.bar(store_stats, x='StoreType', y='Sales', title='Average Sales by Store Type', color='Sales', color_continuous_scale='magma')
            fig4.add_hline(y=overall_mean_sales, line_dash='dash', line_color='red')
            st.plotly_chart(fig4, width='stretch')
            
        # 5. Assortment Type Insights
        with col_d:
            st.subheader("📦 5. Assortment Type Impact")
            assort_stats = df.groupby('Assortment')['Sales'].mean().reset_index()
            fig5 = px.bar(assort_stats, x='Assortment', y='Sales', title='Average Sales by Assortment Type', color='Sales', color_continuous_scale='plasma')
            fig5.add_hline(y=overall_mean_sales, line_dash='dash', line_color='red')
            st.plotly_chart(fig5, width='stretch')
            
        # 6. Holiday Impacts
        st.subheader("🏫 6. State vs School Holiday Impacts")
        col_e, col_f = st.columns(2)
        with col_e:
            hol_stats = df.groupby('StateHoliday')['Sales'].mean().reset_index()
            # Map state holiday to string if it isn't already to handle 0 and '0'
            hol_stats['StateHoliday'] = hol_stats['StateHoliday'].astype(str)
            fig6 = px.bar(hol_stats, x='StateHoliday', y='Sales', title='Impact of State Holidays (a=Public, b=Easter, c=Christmas)', color='StateHoliday', color_discrete_sequence=px.colors.sequential.Reds_r)
            st.plotly_chart(fig6, width='stretch')
        
        with col_f:
            sh_stats = df.groupby('SchoolHoliday')['Sales'].mean().reset_index()
            sh_stats['SchoolHoliday'] = sh_stats['SchoolHoliday'].map({0: 'No', 1: 'Yes'})
            fig7 = px.bar(sh_stats, x='SchoolHoliday', y='Sales', title='Impact of School Holidays', color='SchoolHoliday', color_discrete_sequence=['#2ecc71', '#27ae60'])
            st.plotly_chart(fig7, width='stretch')

        col_g, col_h = st.columns(2)
        
        # 7. Combined Effect of School Holiday and Promo
        with col_g:
            st.subheader("🎓 × 🏷️ 7. Combined Effect of Holiday & Promo")
            combo_stats = df.groupby(['SchoolHoliday', 'Promo'])['Sales'].mean().reset_index()
            # Format labels
            combo_stats['Condition'] = combo_stats.apply(lambda row: f"Promo: {'Yes' if row['Promo']==1 else 'No'} | School Holiday: {'Yes' if row['SchoolHoliday']==1 else 'No'}", axis=1)
            
            fig8 = px.bar(combo_stats, x='Condition', y='Sales', color='Promo',
                               title='Average Sales by Promo and School Holiday Combination',
                               text=combo_stats['Sales'].apply(lambda x: f"${x:,.0f}"),
                               color_continuous_scale='Reds')
            st.plotly_chart(fig8, width='stretch')

        # 8. Month-wise Pattern of School Holidays and Orders
        with col_h:
            st.subheader("📆 8. School Holiday Pattern by Month")
            monthly_sh = df[df['SchoolHoliday'] == 1].groupby(['Month', 'Month_Name'])['Sales'].agg(total_holidays='count', avg_sales='mean').reset_index()
            monthly_sh = monthly_sh.sort_values('Month')
            
            fig9 = make_subplots(specs=[[{"secondary_y": True}]])
            fig9.add_trace(go.Bar(x=monthly_sh['Month_Name'], y=monthly_sh['total_holidays'], name='Holiday Days', marker_color='lightblue'), secondary_y=False)
            fig9.add_trace(go.Scatter(x=monthly_sh['Month_Name'], y=monthly_sh['avg_sales'], name='Avg Sales (During Holiday)', mode='lines+markers', line=dict(color='red')), secondary_y=True)
            fig9.update_layout(title_text='Month-wise School Holidays and their Effect on Sales')
            fig9.update_yaxes(title_text="Total Holiday Days", secondary_y=False)
            fig9.update_yaxes(title_text="Average Sales", secondary_y=True)
            st.plotly_chart(fig9, width='stretch')

        st.markdown("---")
        
        # 9. Month-wise Pattern of Promotions
        st.subheader("📆 9. Month-wise Promotion Pattern and Effectiveness")
        st.markdown("Are promotions run evenly throughout the year? And how effective are they during those months?")
        monthly_promo = df[df['Promo'] == 1].groupby(['Month', 'Month_Name'])['Sales'].agg(total_promos='count', avg_sales='mean').reset_index()
        monthly_promo = monthly_promo.sort_values('Month')
        
        fig10 = make_subplots(specs=[[{"secondary_y": True}]])
        fig10.add_trace(go.Bar(x=monthly_promo['Month_Name'], y=monthly_promo['total_promos'], name='Promo Days', marker_color='orange'), secondary_y=False)
        fig10.add_trace(go.Scatter(x=monthly_promo['Month_Name'], y=monthly_promo['avg_sales'], name='Avg Sales (During Promo)', mode='lines+markers', line=dict(color='darkgreen')), secondary_y=True)
        fig10.update_layout(title_text='Month-wise Promotion Pattern and its Effect on Sales')
        fig10.update_yaxes(title_text="Total Number of Promo Days", secondary_y=False)
        fig10.update_yaxes(title_text="Average Sales during Promos", secondary_y=True)
        st.plotly_chart(fig10, width='stretch')


        st.markdown("---")
        
        # 10. Feature Correlation Heatmap
        st.subheader("🔥 10. Feature Correlation Heatmap")
        st.markdown("How do different numerical features correlate with Sales and with each other?")
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        cols_to_drop = ['Store', 'DayOfWeek', 'Year', 'Month', 'Promo2SinceYear', 'Promo2SinceWeek']
        numeric_df = numeric_df.drop(columns=[col for col in cols_to_drop if col in numeric_df.columns], errors='ignore')
        
        # Calc correlation
        corr_matrix = numeric_df.corr().round(2)
        
        fig11 = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                          title="Correlation Matrix of Numerical Features")
        st.plotly_chart(fig11, width='stretch')

    else:
        st.info("Insights cannot be generated because the raw dataset failed to load.")

# ==========================================================
# PAGE 3: FILTERED INSIGHTS
# ==========================================================
elif page == "🔍 Filtered Insights":
    st.title("🔍 Filtered Insights")
    st.markdown("Slice and dice the data using multiple parameters to find customized insights.")
    
    if raw_data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        
        # Get unique values for filters
        store_types = sorted(df['StoreType'].dropna().unique().tolist())
        assortments = sorted(df['Assortment'].dropna().unique().tolist())
        months = sorted(df['Month'].dropna().unique().tolist())
        days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        selected_store_type = st.sidebar.multiselect("Select Store Type", options=store_types, default=store_types)
        selected_assortment = st.sidebar.multiselect("Select Assortment", options=assortments, default=assortments)
        selected_school_hol = st.sidebar.radio("School Holiday", options=["All", "Yes", "No"])
        selected_months = st.sidebar.multiselect("Select Months", options=months, default=months, format_func=lambda x: pd.to_datetime(str(x), format='%m').month_name())
        selected_days = st.sidebar.multiselect("Select Days of Week", options=days_order, default=days_order)
        exclude_zero_sales = st.sidebar.checkbox("Exclude Zero Sales (Closed Days)", value=True)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_store_type:
            filtered_df = filtered_df[filtered_df['StoreType'].isin(selected_store_type)]
        if selected_assortment:
            filtered_df = filtered_df[filtered_df['Assortment'].isin(selected_assortment)]
        if selected_months:
            filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]
        if selected_days:
            filtered_df = filtered_df[filtered_df['DayName'].isin(selected_days)]
            
        if selected_school_hol == "Yes":
            filtered_df = filtered_df[filtered_df['SchoolHoliday'] == 1]
        elif selected_school_hol == "No":
            filtered_df = filtered_df[filtered_df['SchoolHoliday'] == 0]
            
        if exclude_zero_sales:
            filtered_df = filtered_df[filtered_df['Sales'] > 0]
            
        if len(filtered_df) == 0:
            st.warning("No data available for the selected filters.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stores in Filter", filtered_df['Store'].nunique())
            with col2:
                st.metric("Average Sales", f"${filtered_df['Sales'].mean():,.2f}")
            with col3:
                st.metric("Total Records", f"{len(filtered_df):,}")
            
            # Visualization 1: Sales Distribution
            st.subheader("📉 Sales Distribution for Filtered Data")
            fig_dist = px.histogram(filtered_df, x='Sales', nbins=50, 
                                    title='Distribution of Sales',
                                    color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_dist, width='stretch')
            
            # Visualization 2: Promo Performance within Filter
            st.subheader("🏷️ Promotion Effectiveness in Filtered Segment")
            promo_filtered = filtered_df.groupby('Promo')['Sales'].mean().reset_index()
            promo_filtered['Promo_Label'] = promo_filtered['Promo'].map({0: 'No Promo', 1: 'Promo Active'})
            
            fig_promo2 = px.bar(promo_filtered, x='Promo_Label', y='Sales', color='Promo_Label',
                                title='Average Sales (Promo vs No Promo)',
                                text=promo_filtered['Sales'].apply(lambda x: f"${x:,.0f}"))
            st.plotly_chart(fig_promo2, width='stretch')
            
    else:
        st.info("Insights cannot be generated because the raw dataset failed to load.")

# ==========================================================
# PAGE 4: INVENTORY MANAGEMENT
# ==========================================================
elif page == "📦 Inventory Management":
    st.title("📦 Inventory Management System")
    st.markdown("Track and manage store inventory levels. Keep an eye out for low stock!")
    
    INVENTORY_FILE = 'datasets/inventory.csv'
    
    # Initialize inventory file if missing
    if not os.path.exists(INVENTORY_FILE):
        default_data = {
            "Item_ID": [1001, 1002, 1003],
            "Item_Name": ["Premium Shampoo", "Organic Soap", "Pain Reliever"],
            "Category": ["Health & Beauty", "Health & Beauty", "Medicine"],
            "Stock": [150, 40, 200],
            "Reorder_Level": [50, 60, 100],
            "Unit_Price": [5.99, 3.49, 8.99]
        }
        pd.DataFrame(default_data).to_csv(INVENTORY_FILE, index=False)
        
    def load_inventory():
        return pd.read_csv(INVENTORY_FILE)
        
    inv_df = load_inventory()
    
    # Overview Metrics
    total_items = len(inv_df)
    low_stock_items = len(inv_df[inv_df['Stock'] <= inv_df['Reorder_Level']])
    if total_items > 0:
        total_value = (inv_df['Stock'] * inv_df['Unit_Price']).sum()
    else:
        total_value = 0.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Unique Items</div><div class="metric-value" style="color:#3498db;">{total_items}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Low Stock Alerts</div><div class="metric-value" style="color:#e74c3c;">{low_stock_items}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Inventory Value</div><div class="metric-value" style="color:#2ecc71;">${total_value:,.2f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("### 📋 Current Inventory List")
    st.markdown("Edit directly in the table below to update stock levels, prices, or names.")
    
    edited_df = st.data_editor(
        inv_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Item_ID": st.column_config.NumberColumn("Item ID", disabled=True),
            "Stock": st.column_config.NumberColumn("Current Stock", min_value=0),
            "Reorder_Level": st.column_config.NumberColumn("Reorder Level", min_value=0),
            "Unit_Price": st.column_config.NumberColumn("Price ($)", min_value=0.0, format="$%.2f")
        }
    )
    
    if st.button("💾 Save Inventory Changes"):
        try:
            edited_df.to_csv(INVENTORY_FILE, index=False)
            st.success("Inventory updated successfully!")
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to save changes: {e}")
            
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Determine what items need reordering
        st.markdown("### ⚠️ Low Stock Action Items")
        action_df = edited_df[edited_df['Stock'] <= edited_df['Reorder_Level']].copy()
        if len(action_df) > 0:
            action_df['Deficit'] = action_df['Reorder_Level'] - action_df['Stock']
            st.warning(f"You have {len(action_df)} item(s) currently under or at the recommended reorder level.")
            st.dataframe(action_df[['Item_ID', 'Item_Name', 'Stock', 'Reorder_Level']], use_container_width=True, hide_index=True)
        else:
            st.success("All items are currently well-stocked. No immediate action required.")

    with col_right:
        st.markdown("### ➕ Add New Item")
        with st.form("add_item_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                new_name = st.text_input("Item Name")
                new_cat = st.selectbox("Category", ["Health & Beauty", "Medicine", "Personal Care", "Household", "Baby Care", "Other"])
                new_price = st.number_input("Unit Price ($)", min_value=0.0, step=0.5, value=1.99)
            with col_b:
                new_stock = st.number_input("Initial Stock", min_value=0, value=0)
                new_reorder = st.number_input("Reorder Level", min_value=0, value=10)
                
            submitted = st.form_submit_button("Add Item")
            
            if submitted:
                if new_name.strip() == "":
                    st.error("Item name cannot be empty.")
                else:
                    new_id = int(inv_df['Item_ID'].max() + 1) if not inv_df.empty else 1000
                    new_row = pd.DataFrame({
                        "Item_ID": [new_id],
                        "Item_Name": [new_name],
                        "Category": [new_cat],
                        "Stock": [new_stock],
                        "Reorder_Level": [new_reorder],
                        "Unit_Price": [new_price]
                    })
                    updated_df = pd.concat([inv_df, new_row], ignore_index=True)
                    updated_df.to_csv(INVENTORY_FILE, index=False)
                    st.success(f"Added {new_name} with ID {new_id}!")
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
