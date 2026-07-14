import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

print("Loading BigMart Sales Prediction Pipeline...")

try:
    # 1. Load your local dataset 
    df = pd.read_csv("bigmart_sales.csv")
    print(f"Dataset Dimensions: {df.shape}")
    
    # Fill basic missing values to prevent execution crashes
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 2. Fix the Pandas4Warning by explicitly passing both 'object' and 'string'
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    
    # Transform text labels into numeric fields
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 3. Define Features (X) and Target Label (y)
    # Target variable is assumed to be 'Item_Outlet_Sales'
    X = df.drop(columns=['Item_Outlet_Sales'])
    y = df['Item_Outlet_Sales']
    
    # Split dataset into Train and Test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train the Model
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Model Evaluation Output
    y_pred = model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Regression Performance Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score:           {r2:.4f} (approx. {r2*100:.1f}% variance explained)")
    
except FileNotFoundError:
    print("\n🚨 Error: 'bigmart_sales.csv' dataset file could not be found in this workspace directory folder.")
except Exception as e:
    print(f"\n❌ A runtime processing error occurred: {str(e)}")
