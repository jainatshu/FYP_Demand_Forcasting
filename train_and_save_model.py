import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def main():
    print("Loading data...")
    train_df = pd.read_csv('processed_train.csv')

    print("Preparing features...")
    top_features = [
        'CompetitionDistance', 'Promo', 'CompetitionOpenSinceMonth', 
        'CompetitionOpenSinceYear', 'StoreType', 'Promo2SinceYear', 
        'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 
        'Promo2SinceWeek', 'Assortment', 'SchoolHoliday'
    ]

    X = train_df[top_features].fillna(0)
    y = train_df['Sales'].fillna(0)

    # Train-test split (from the notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("Saving model to rf_model.pkl...")
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()
