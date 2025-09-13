from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('FinAnalysis') \
    .getOrCreate()


txn_df = spark.read.csv(
    "gs://fin_data_bucket/transactions_dataset.csv",  # Google Cloud Storage path
    header=True,
    inferSchema=True
)
dataset_columns = txn_df.columns



# Check for null values
from pyspark.sql.functions import col, sum
null_values = txn_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in dataset_columns])


# Check Data distribution
total_count = txn_df.count()
fraud_count = txn_df.groupBy("is_fraud").count()
print("Number of partitions : ", txn_df.rdd.getNumPartitions())


# # Data Preprocessing

from pyspark.sql.functions import col
cols_to_drop = [
    "transaction_id",
    "customer_id",
    "card_number",
    "timestamp",
    "device_fingerprint",
    "ip_address",
    "velocity_last_hour"
]

txn_df = txn_df.drop(*cols_to_drop)
txn_df = txn_df.withColumn("is_fraud", col("is_fraud").cast("integer"))


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler

# Identifying columns
string_cols = [field.name for field in txn_df.schema.fields if field.dataType.typeName() == "string"]
boolean_cols = [field.name for field in txn_df.schema.fields if field.dataType.typeName() == "boolean"]
numeric_cols = [field.name for field in txn_df.schema.fields if field.dataType.typeName in ["integer", "double", "long"]]
timestamp_cols = [field.name for field in txn_df.schema.fields if field.dataType.typeName == "timestamp"]




indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") for col in string_cols]


feature_cols = [col+"_idx" for col in string_cols] + boolean_cols + numeric_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")


# Building the Pipeline
pipeline = Pipeline(stages= indexers + [assembler , scaler])

pipeline_model = pipeline.fit(txn_df)
preprocessed_df = pipeline_model.transform(txn_df)


train_df, test_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)


from pyspark.ml.classification import LogisticRegression

logistic_regression = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud")
model = logistic_regression.fit(train_df)



from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud")
auc = evaluator.evaluate(predictions)
print(f"Test AUC : {auc}")



from pyspark.sql.functions import col

conf_df = predictions.select(
    col("is_fraud").alias("actual"),
    col("prediction").cast("integer").alias("predicted")
)

conf_matrix = conf_df.groupBy("actual","predicted").count().orderBy("actual", "predicted")
conf_matrix.show()


# Verify distributed execution
print(f"Spark Context: {spark.sparkContext}")
print(f"Master: {spark.sparkContext.master}")
print(f"App Name: {spark.sparkContext.appName}")
print(f"Number of Executors: {len(spark.sparkContext.statusTracker().getExecutorInfos()) - 1}")  # -1 for driver
print(f"Data Partitions: {txn_df.rdd.getNumPartitions()}")

# Show partition distribution
partition_sizes = txn_df.rdd.glom().map(len).collect()
print(f"Rows per partition: {partition_sizes}")

