import pandas as pd

print("Initializing Pure Python Movie Analytics Pipeline...")

# 1. Generate matrix data records and save to local workspace
data = {
    "MovieID": [1, 1, 1, 2, 2, 3, 3, 3, 3],
    "UserID": [101, 102, 103, 101, 102, 101, 102, 103, 104],
    "Rating": [5, 4, 3, 5, 4, 2, 3, 4, 5],
    "Timestamp": [111, 112, 113, 114, 115, 116, 117, 118, 119]
}

df = pd.DataFrame(data)
df.to_csv("movies.csv", index=False)

print("\n--- Input Dataset Matrix Sample ---")
print(df.to_string(index=False))

# 2. Compute compound evaluation averages using standard Pandas grouping
# This replaces the PySpark groupBy/agg mechanics natively in Python
result = df.groupby("MovieID")["Rating"].mean().reset_index()
result.rename(columns={"Rating": "Average_Rating"}, inplace=True)

print("\n=============================================")
print("          AVERAGE RATING OF MOVIES           ")
print("=============================================")
print(result.to_string(index=False))
print("=============================================")
