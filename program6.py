import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Global Fix: Suppress warning logs from cluttering the Streamlit console text area
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

print("Loading Bike Trip Classification Pipeline...")

try:
    # 1. Load your local dataset 
    df = pd.read_csv("trip_history.csv")
    
    print("\nFirst 5 Records")
    print(df.head())
    
    print("\nColumns")
    print(df.columns)
    
    # 2. Complete String Type Check: Safely filter columns regardless of 'object' vs 'str' setup
    categorical_cols = [c for c in df.columns if df[c].dtype in ['object', 'str', 'O'] or isinstance(df[c].dtype, pd.StringDtype)]
    
    # Encode categorical text values to numbers natively
    le = LabelEncoder()
    for col in categorical_cols:
        # Exclude the target variable from encoding if handled manually
        if col != 'UserType':
            df[col] = le.fit_transform(df[col].astype(str))
            
    # Ensure target variable UserType is encoded if it is text data
    if df['UserType'].dtype in ['object', 'str', 'O']:
        df['UserType'] = le.fit_transform(df['UserType'].astype(str))
        
    # 3. Define Features (X) and Target Label (y)
    X = df.drop(columns=['UserType'])
    y = df['UserType']
    
    # Split dataset into Train and Test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Initialize Classifier and fit models
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Model Evaluation Output
    y_pred = model.predict(X_test)
    
    print(f"\nAccuracy = {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))
    
except FileNotFoundError:
    print("\n🚨 Error: 'trip_history.csv' dataset file could not be found in this workspace directory folder.")
except Exception as e:
    print(f"\n❌ A runtime processing error occurred: {str(e)}")
