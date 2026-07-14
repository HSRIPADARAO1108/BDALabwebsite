import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the BigMart Dataset
# (Assuming your dataset file is named 'bigmart_sales.csv')
df = pd.read_csv("bigmart_sales.csv")

print("Dataset Dimensions:", df.shape) # Expected: (8523, 12)

# 2. Handling Missing Values
# BigMart data typically has missing values in 'Item_Weight' and 'Outlet_Size'
if 'Item_Weight' in df.columns:
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())

if 'Outlet_Size' in df.columns:
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

# 3. Feature Engineering & Cleaning
# Standardizing text anomalies (BigMart often has 'LF', 'low fat' alongside 'Low Fat')
if 'Item_Fat_Content' in df.columns:
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

# 4. Encoding Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    # Drop unique identifiers like Item_Identifier to avoid overfitting, encode others
    if col in ['Item_Identifier', 'Outlet_Identifier']:
        df = df.drop(columns=[col])
    else:
        df[col] = le.fit_transform(df[col])

# 5. Define Features (X) and Target (y)
# Target variable is Item_Outlet_Sales
X = df.drop(columns=['Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model Selection & Training (Random Forest Regressor)
print("\nTraining Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Model Evaluation
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Performance Evaluation ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score:           {r2:.4f} (approx. {r2*100:.1f}% variance explained)")
