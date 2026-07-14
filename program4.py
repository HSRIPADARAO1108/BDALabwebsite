!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark Session
spark = SparkSession.builder.appName("Weather").getOrCreate()

# Load CSV file
df = spark.read.csv("weather.csv", header=True, inferSchema=True)

# Display Dataset
print("Weather Dataset")
df.show()

# Filter records for 2013
weather2013 = df.filter(col("Date").startswith("2013"))

# Find Maximum Snowfall
maxSnow = weather2013.orderBy(col("Snowfall").desc()).first()

print("\nMaximum Snowfall in 2013")
print("Station :", maxSnow["Station"])
print("Date    :", maxSnow["Date"])
print("Snowfall:", maxSnow["Snowfall"])
