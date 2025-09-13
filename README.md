# Transaction Fraud Detection with PySpark MLlib

A machine learning project for detecting fraudulent transactions using distributed computing with Apache Spark and MLlib.

## Overview

This project implements a binary classification model to predict whether financial transactions are fraudulent or legitimate. The solution leverages PySpark for distributed data processing and MLlib for scalable machine learning.

## Dataset

The dataset contains transaction records with the following features:

- Transaction details (merchant, amount, currency, country, city)
- Card information (card type, presence)
- Temporal features (timestamp, transaction hour, weekend flag)
- Risk indicators (high-risk merchant, distance from home)
- Device and channel information
- Target variable: `is_fraud` (binary classification)

**Dataset Statistics:**

- Total transactions: ~7.5M
- Fraud cases: ~1.5M (20%)
- Non-fraud cases: ~6M (80%)

## Technology Stack

- **Apache Spark**: Distributed data processing
- **PySpark**: Python API for Spark
- **MLlib**: Spark's machine learning library
- **Python**: Primary programming language
- **Jupyter Notebook**: Development environment

## Project Structure

``` bash
TransactionAnalysis/
├── analysis.ipynb          # Main notebook with complete pipeline
├── README.md              # Project documentation
└── data/                  # Dataset files (not included in repo)
```

## Methodology

### 1. Data Preprocessing

- **Feature Selection**: Removed high-cardinality columns (transaction IDs, timestamps, device fingerprints)
- **Data Cleaning**: Handled missing values and data types
- **Feature Engineering**:
  - String indexing for categorical variables
  - One-hot encoding for multi-class categories
  - Feature vector assembly using `VectorAssembler`
  - Feature scaling with `StandardScaler`

### 2. Model Development

- **Algorithm**: Logistic Regression (binary classification)
- **Train-Test Split**: 80/20 split
- **Distributed Training**: Leveraged Spark's parallel processing capabilities

### 3. Model Evaluation

- **Primary Metric**: AUC-ROC = 0.845
- **Confusion Matrix Analysis**: Evaluated precision, recall, and F1-score
- **Class Imbalance Handling**: Adjusted prediction thresholds

## Key Results

- **AUC Score**: 0.845 (good discrimination between fraud/non-fraud)
- **Model Performance**: Conservative approach with high precision, moderate recall
- **Scalability**: Successfully processed 3GB+ dataset using distributed computing

## Implementation Highlights

### Distributed Data Processing

```python
# Spark session configuration
spark = SparkSession.builder \
    .appName('FinAnalysis') \
    .master('local[*]') \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Data partitioning for parallel processing
txn_df = spark.read.csv(path, header=True, inferSchema=True)
print(f"Data distributed across {txn_df.rdd.getNumPartitions()} partitions")
```

### ML Pipeline

```python
# Feature preprocessing pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
pipeline_model = pipeline.fit(txn_df)
preprocessed_df = pipeline_model.transform(txn_df)

# Model training and evaluation
lr = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud")
model = lr.fit(train_df)
predictions = model.transform(test_df)
```

## Challenges and Solutions

1. **Memory Management**:
   - Issue: OutOfMemoryError with large dataset
   - Solution: Increased driver memory and optimized data partitioning

2. **Class Imbalance**:
   - Issue: 80/20 imbalanced dataset leading to prediction bias
   - Solution: Threshold tuning and evaluation metric selection

3. **High Cardinality Features**:
   - Issue: Features with too many unique values
   - Solution: Feature selection based on cardinality analysis

## Usage

1. **Setup Environment**:

   ```bash
   pip install pyspark
   ```

2. **Run Analysis**:

   ```python
   # Open analysis.ipynb in Jupyter Notebook
   # Execute cells sequentially
   ```

3. **Model Training**:

   ```python
   # The notebook includes complete pipeline from data loading to evaluation
   ```

## Performance Considerations

- **Scalability**: Designed for distributed processing across multiple nodes
- **Memory Optimization**: Configured for optimal resource utilization
- **Data Partitioning**: Balanced partitions for efficient parallel processing

## Future Improvements

- Implement advanced algorithms (Random Forest, Gradient Boosting)
- Deploy model using Spark Streaming for real-time fraud detection
- Integrate feature selection techniques
- Implement cross-validation and hyperparameter tuning

## License

This project is for educational and research purposes.

## Contact

For questions or collaboration opportunities, please reach out through GitHub.
