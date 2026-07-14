import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris.csv")

print("="*50)
print("IRIS DATASET")
print("="*50)

# Shape
print("\nNumber of Rows :", df.shape[0])
print("Number of Columns :", df.shape[1])

# Features
print("\nFeatures:")
print(df.columns.tolist())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Numeric and Nominal Columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
nominal_cols = df.select_dtypes(include=['object']).columns

print("\nNumeric Features:")
print(list(numeric_cols))

print("\nNominal Features:")
print(list(nominal_cols))

# Summary Statistics
print("\nSummary Statistics")
print(df.describe())

# Mean
print("\nMean")
print(df[numeric_cols].mean())

# Minimum
print("\nMinimum")
print(df[numeric_cols].min())

# Maximum
print("\nMaximum")
print(df[numeric_cols].max())

# Range
print("\nRange")
print(df[numeric_cols].max() - df[numeric_cols].min())

# Standard Deviation
print("\nStandard Deviation")
print(df[numeric_cols].std())

# Variance
print("\nVariance")
print(df[numeric_cols].var())

# Percentiles
print("\nPercentiles")
print(df[numeric_cols].quantile([0.25,0.5,0.75]))

# Histograms
df[numeric_cols].hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# Combined Boxplot
plt.figure(figsize=(8,6))
df[numeric_cols].boxplot()
plt.title("Combined Boxplot of Iris Dataset")
plt.show()
