
import pandas as pd

data = {
    "MovieID":[1,1,1,2,2,3,3,3,3],
    "UserID":[101,102,103,101,102,101,102,103,104],
    "Rating":[5,4,3,5,4,2,3,4,5],
    "Timestamp":[111,112,113,114,115,116,117,118,119]
}

df = pd.DataFrame(data)
df.to_csv("movies.csv", index=False)

df

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Create Spark Session
spark = SparkSession.builder.appName("MovieRatings").getOrCreate()

# Read CSV
movies = spark.read.csv("movies.csv", header=True, inferSchema=True)

print("Input Dataset")
movies.show()

# Average rating of each movie
result = movies.groupBy("MovieID").agg(avg("Rating").alias("Average_Rating"))

print("Average Rating of Movies")
result.show()
