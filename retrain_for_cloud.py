import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import sklearn
import os

print(f"Using scikit-learn {sklearn.__version__}")

print("Loading data (with downcasting)...")
train_df = pd.read_csv('processed_train.csv')

# Sample down to 30% to reduce memory during training
train_df = train_df.sample(frac=0.3, random_state=42)
print(f"Training on {len(train_df)} rows (30% sample)")

top_features = [
    'CompetitionDistance', 'Promo', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'StoreType', 'Promo2SinceYear',
    'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6',
    'Promo2SinceWeek', 'Assortment', 'SchoolHoliday'
]

X = train_df[top_features].fillna(0)
y = train_df['Sales'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use fewer trees and lower depth for smaller RAM footprint on 512MB free tiers
print("Training lightweight Random Forest (n_estimators=12, max_depth=20)...")
rf_model = RandomForestRegressor(
    n_estimators=12,
    max_depth=20,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

print("Saving compressed model with max compression...")
joblib.dump(rf_model, 'rf_model_compressed.pkl.gz', compress=('gzip', 9))
size = os.path.getsize('rf_model_compressed.pkl.gz') / 1e6
print(f"Saved rf_model_compressed.pkl.gz ({size:.1f} MB)")
print("Done!")
