# Databricks notebook source
print("Databricks Netflix Project Started")

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/netflix_data/netflix_user_behavior_dataset.csv",
    header=True,
    inferSchema=True
)

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df

# COMMAND ----------

df.count()

# COMMAND ----------

df.columns

# COMMAND ----------

# DBTITLE 1,Check Missing Values
from pyspark.sql.functions import col, sum

df.select(
    [
        sum(col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]
).show()

# COMMAND ----------

# DBTITLE 1,Remove Missing Values
df_clean = df.dropna()

# COMMAND ----------

df_clean.count()

# COMMAND ----------

# DBTITLE 1,Remove Duplicate Rows
df_clean = df_clean.dropDuplicates()

# COMMAND ----------

df_clean.count()

# COMMAND ----------

# DBTITLE 1,Check Data Types
df_clean.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col

df_clean = df_clean.withColumn(
    "Age",
    col("Age").cast("integer")
)

# COMMAND ----------

# DBTITLE 1,Rename Columns
df_clean = df_clean.withColumnRenamed(
    "Watch Time",
    "Watch_Time"
)

# COMMAND ----------

# DBTITLE 1,Final Clean Data Preview
df_clean.show(10)

# COMMAND ----------

# DBTITLE 1,Save Clean Dataset
df_clean.write.mode("overwrite").csv(
"/Volumes/workspace/default/netflix_data/cleaned_netflix"
)

# COMMAND ----------

# DBTITLE 1,Create Temporary SQL Table
df_clean.createOrReplaceTempView("netflix_users")

# COMMAND ----------

# DBTITLE 1,View Data
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM netflix_users
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Total Users Count
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS Total_Users
# MAGIC FROM netflix_users;

# COMMAND ----------

df_clean.columns

# COMMAND ----------

# DBTITLE 1,Age Analysis
# MAGIC %sql
# MAGIC SELECT
# MAGIC AVG(age) AS Average_Age
# MAGIC FROM netflix_users;

# COMMAND ----------

df_clean.createOrReplaceTempView("netflix_users")

# COMMAND ----------

# DBTITLE 1,Total Users
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS Total_Users
# MAGIC FROM netflix_users;

# COMMAND ----------

# DBTITLE 1,Country Wise Users
# MAGIC %sql
# MAGIC SELECT 
# MAGIC country,
# MAGIC COUNT(*) AS Users
# MAGIC FROM netflix_users
# MAGIC GROUP BY country
# MAGIC ORDER BY Users DESC;

# COMMAND ----------

# DBTITLE 1,Favorite Genre Analysis
# MAGIC %sql
# MAGIC SELECT
# MAGIC favorite_genre,
# MAGIC COUNT(*) AS Views
# MAGIC FROM netflix_users
# MAGIC GROUP BY favorite_genre
# MAGIC ORDER BY Views DESC;

# COMMAND ----------

# DBTITLE 1,Subscription Analysis
# MAGIC %sql
# MAGIC SELECT
# MAGIC subscription_type,
# MAGIC COUNT(*) AS Customers
# MAGIC FROM netflix_users
# MAGIC GROUP BY subscription_type;

# COMMAND ----------

# DBTITLE 1,Average Watch Time
# MAGIC %sql
# MAGIC SELECT
# MAGIC AVG(avg_watch_time_minutes) AS Average_Watch_Time
# MAGIC FROM netflix_users;

# COMMAND ----------

# DBTITLE 1,Churn Analysis
# MAGIC %sql
# MAGIC SELECT
# MAGIC churned,
# MAGIC COUNT(*) AS Customers
# MAGIC FROM netflix_users
# MAGIC GROUP BY churned;

# COMMAND ----------

# DBTITLE 1,Device Usage
# MAGIC %sql
# MAGIC SELECT
# MAGIC primary_device,
# MAGIC COUNT(*) AS Users
# MAGIC FROM netflix_users
# MAGIC GROUP BY primary_device
# MAGIC ORDER BY Users DESC;

# COMMAND ----------

# DBTITLE 1,Highest Engagement Users
# MAGIC %sql
# MAGIC SELECT
# MAGIC user_id,
# MAGIC avg_watch_time_minutes,
# MAGIC watch_sessions_per_week,
# MAGIC binge_watch_sessions
# MAGIC FROM netflix_users
# MAGIC ORDER BY avg_watch_time_minutes DESC
# MAGIC LIMIT 10;

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# COMMAND ----------

# DBTITLE 1,Select Features
features = [
    "avg_watch_time_minutes",
    "watch_sessions_per_week",
    "binge_watch_sessions",
    "completion_rate",
    "rating_given"
]

# COMMAND ----------

# DBTITLE 1,Convert Columns into Feature Vector
assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df_ml = assembler.transform(df_clean)

df_ml.select("features").show(5)

# COMMAND ----------

# DBTITLE 1,Train K-Means Model
kmeans = KMeans(
    k=3,
    seed=42,
    featuresCol="features",
    predictionCol="cluster"
)

model = kmeans.fit(df_ml)

df_clustered = model.transform(df_ml)

# COMMAND ----------

# DBTITLE 1,View Segments
df_clustered.select(
    "user_id",
    "avg_watch_time_minutes",
    "cluster"
).show(10)

# COMMAND ----------

# DBTITLE 1,Count Users in Each Segment
df_clustered.groupBy("cluster").count().show()

# COMMAND ----------

# DBTITLE 1,Average values by cluster
from pyspark.sql.functions import avg

df_clustered.groupBy("cluster").agg(
    avg("avg_watch_time_minutes").alias("Avg_Watch_Time"),
    avg("watch_sessions_per_week").alias("Avg_Sessions"),
    avg("binge_watch_sessions").alias("Avg_Binge"),
    avg("completion_rate").alias("Avg_Completion"),
    avg("rating_given").alias("Avg_Rating")
).show()

# COMMAND ----------

# DBTITLE 1,Add Segment Name
from pyspark.sql.functions import when

df_final = df_clustered.withColumn(
    "User_Segment",
    when(df_clustered.cluster == 0, "Low Engagement")
    .when(df_clustered.cluster == 1, "Regular Users")
    .otherwise("Heavy Viewers")
)

df_final.show()

# COMMAND ----------

# DBTITLE 1,Check Final Data
df_final.select(
    "user_id",
    "country",
    "subscription_type",
    "avg_watch_time_minutes",
    "User_Segment"
).show(10)

# COMMAND ----------

display(
    df_final.groupBy("User_Segment")
    .count()
)

# COMMAND ----------

df_final.groupBy("User_Segment").count().show()

# COMMAND ----------

# DBTITLE 1,User Segmentation Analysis Chart
import matplotlib.pyplot as plt

segment_count = (
    df_final
    .groupBy("User_Segment")
    .count()
    .toPandas()
)

plt.figure(figsize=(8,5))

plt.bar(
    segment_count["User_Segment"],
    segment_count["count"]
)

plt.xlabel("User Segment")
plt.ylabel("Number of Users")
plt.title("Netflix User Segmentation")

plt.show()

# COMMAND ----------

# DBTITLE 1,Churn Analysis Chart
import matplotlib.pyplot as plt

churn_data = (
    df_final
    .groupBy("churned")
    .count()
    .toPandas()
)

plt.figure(figsize=(6,4))

plt.bar(
    churn_data["churned"].astype(str),
    churn_data["count"]
)

plt.xlabel("Churn Status")
plt.ylabel("Customers")
plt.title("Netflix Customer Churn Analysis")

plt.show()

# COMMAND ----------

# DBTITLE 1,Subscription Type Chart
subscription_data = (
    df_final
    .groupBy("subscription_type")
    .count()
    .toPandas()
)

plt.figure(figsize=(7,4))

plt.bar(
    subscription_data["subscription_type"],
    subscription_data["count"]
)

plt.xlabel("Subscription Type")
plt.ylabel("Users")
plt.title("Subscription Distribution")

plt.show()

# COMMAND ----------

# DBTITLE 1,Genre Preference Analysis chart
import matplotlib.pyplot as plt

genre_count = (
    df_final
    .groupBy("favorite_genre")
    .count()
    .toPandas()
)

plt.figure(figsize=(8,5))

plt.bar(
    genre_count["favorite_genre"],
    genre_count["count"]
)

plt.xlabel("Favorite Genre")
plt.ylabel("Number of Users")
plt.title("Genre Preference Analysis")

plt.xticks(rotation=45)

plt.show()

# COMMAND ----------

# DBTITLE 1,Genre Wise Average Watch Time Chart
import matplotlib.pyplot as plt

watch_data = (
    df_final
    .groupBy("favorite_genre")
    .avg("avg_watch_time_minutes")
    .toPandas()
)

plt.figure(figsize=(8,5))

plt.bar(
    watch_data["favorite_genre"],
    watch_data["avg(avg_watch_time_minutes)"]
)

plt.xlabel("Favorite Genre")
plt.ylabel("Average Watch Time (Minutes)")
plt.title("Genre Wise Average Watch Time")

plt.xticks(rotation=45)

plt.show()

# COMMAND ----------

# DBTITLE 1,Device Usage Analysis Chart
import matplotlib.pyplot as plt

device_count = (
    df_final
    .groupBy("primary_device")
    .count()
    .toPandas()
)

plt.figure(figsize=(7,5))

plt.bar(
    device_count["primary_device"],
    device_count["count"]
)

plt.xlabel("Primary Device")
plt.ylabel("Number of Users")
plt.title("Device Usage Analysis")

plt.show()

# COMMAND ----------

# DBTITLE 1,Subscription Type Analysis Chart
import matplotlib.pyplot as plt

subscription_count = (
    df_final
    .groupBy("subscription_type")
    .count()
    .toPandas()
)

plt.figure(figsize=(7,5))

plt.bar(
    subscription_count["subscription_type"],
    subscription_count["count"]
)

plt.xlabel("Subscription Type")
plt.ylabel("Number of Users")
plt.title("Subscription Type Analysis")

plt.show()

# COMMAND ----------

# DBTITLE 1,Average Watch Time by Subscription
import matplotlib.pyplot as plt

watch_time = (
    df_final
    .groupBy("subscription_type")
    .avg("avg_watch_time_minutes")
    .toPandas()
)

plt.figure(figsize=(7,5))

plt.bar(
    watch_time["subscription_type"],
    watch_time["avg(avg_watch_time_minutes)"]
)

plt.xlabel("Subscription Type")
plt.ylabel("Average Watch Time (Minutes)")
plt.title("Average Watch Time by Subscription")

plt.show()

# COMMAND ----------

df_powerbi = df_final.drop("features")

# COMMAND ----------

df_powerbi.write \
.mode("overwrite") \
.option("header", True) \
.csv("/Volumes/workspace/default/netflix_data/netflix_powerbi_output")

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/workspace/default/netflix_data/netflix_powerbi_output"))

# COMMAND ----------

# MAGIC %whos

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# COMMAND ----------

features = [
    "avg_watch_time_minutes",
    "watch_sessions_per_week",
    "binge_watch_sessions",
    "completion_rate",
    "rating_given"
]

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/netflix_data/netflix_user_behavior_dataset.csv",
    header=True,
    inferSchema=True
)

df.show(5)

# COMMAND ----------

df_clean = df.dropna().dropDuplicates()

df_clean.show(5)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

features = [
    "avg_watch_time_minutes",
    "watch_sessions_per_week",
    "binge_watch_sessions",
    "completion_rate",
    "rating_given"
]

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df_ml = assembler.transform(df_clean)


kmeans = KMeans(
    k=3,
    seed=42,
    featuresCol="features",
    predictionCol="cluster"
)

model = kmeans.fit(df_ml)

df_clustered = model.transform(df_ml)

# COMMAND ----------

from pyspark.sql.functions import when

df_final = df_clustered.withColumn(
    "User_Segment",
    when(df_clustered.cluster == 0, "Low Engagement")
    .when(df_clustered.cluster == 1, "Regular Users")
    .otherwise("Heavy Viewers")
)

df_final.show(5)

# COMMAND ----------

df_powerbi = df_final.drop("features")

# COMMAND ----------

df_powerbi = df_powerbi.select(
    "user_id",
    "Age",
    "gender",
    "country",
    "subscription_type",
    "favorite_genre",
    "avg_watch_time_minutes",
    "watch_sessions_per_week",
    "churned",
    "User_Segment"
)

df_powerbi.show(5)

# COMMAND ----------

df_powerbi = df_powerbi.select(
    "user_id",
    "Age",
    "gender",
    "country",
    "subscription_type",
    "favorite_genre",
    "avg_watch_time_minutes",
    "watch_sessions_per_week",
    "churned",
    "User_Segment"
)

df_powerbi.show(5)

# COMMAND ----------

df_powerbi.coalesce(1).write \
.mode("overwrite") \
.option("header", True) \
.csv("/Volumes/workspace/default/netflix_data/netflix_dashboard_final")

# COMMAND ----------

display(
    dbutils.fs.ls("/Volumes/workspace/default/netflix_data/netflix_dashboard_final")
)