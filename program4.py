import pandas as pd

print("Initializing Pure Python Weather Data Pipeline...")

try:
    # 1. Load the local CSV dataset using standard Pandas
    df = pd.read_csv("weather.csv")
    
    print("\n--- Weather Dataset Sample View ---")
    print(df.head(10).to_string(index=False))
    
    # 2. Filter records where the Date column starts with '2013'
    # Converted to string first to prevent missing data type errors
    weather2013 = df[df['Date'].astype(str).str.startswith('2013')]
    
    # 3. Find the Maximum Snowfall record
    if not weather2013.empty:
        # Sort values descending by Snowfall and select the very first row
        maxSnow = weather2013.sort_values(by='Snowfall', ascending=False).iloc[0]
        
        print("\n=============================================")
        print("          MAXIMUM SNOWFALL IN 2013           ")
        print("=============================================")
        print(f"Station  : {maxSnow['Station']}")
        print(f"Date     : {maxSnow['Date']}")
        print(f"Snowfall : {maxSnow['Snowfall']}")
        print("=============================================")
    else:
        print("\n⚠️ No weather data records found matching the target year 2013.")
        
except FileNotFoundError:
    print("\n🚨 Error: 'weather.csv' dataset file could not be found in this workspace directory folder.")
except Exception as e:
    print(f"\n❌ A runtime processing error occurred: {str(e)}")
