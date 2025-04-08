from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize a SparkSession
spark = SparkSession.builder \
    .appName("Transaction Data Processor") \
    .getOrCreate()

# Specify the path to the CSV file
csv_file_path = "path/to/your/transaction_data.csv"

# Read the CSV file into a DataFrame
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Data Cleaning and Normalization
df = df.dropna()  # Drop rows with null values
df = df.withColumn("transaction_amount", col("transaction_amount").cast("double"))

# Feature Engineering
# Assuming columns 'latitude', 'longitude', 'transaction_amount', 'user_id', 'transaction_date'
df = df.withColumn("frequency", count("user_id").over(Window.partitionBy("user_id")))
df = df.withColumn("geo_location_mismatch", when(
    (col("latitude") != col("expected_latitude")) | (col("longitude") != col("expected_longitude")), 1).otherwise(0))

# Assembling features into a feature vector
assembler = VectorAssembler(
    inputCols=['transaction_amount', 'frequency', 'geo_location_mismatch'],
    outputCol='features'
)
feature_df = assembler.transform(df)

# Applying MLlib Model
# Using Logistic Regression as an example
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(feature_df)
predictions = model.transform(feature_df)

# Alternatively, you could use IsolationForest for anomaly detection
# from pyspark.ml.iforest import IsolationForest
# iso_forest = IsolationForest(featuresCol='features')
# model = iso_forest.fit(feature_df)
# predictions = model.transform(feature_df)

# Output the result to JSON
predictions.select("user_id", "transaction_id", "prediction").write.json("path/to/output/result.json")

# Stop the SparkSession
spark.stop()
