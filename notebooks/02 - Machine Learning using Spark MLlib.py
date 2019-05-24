# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Lab 2: Training a Model using Spark MLlib
# MAGIC Spark MLlib provides a Python, R and Scala API to machine learning algorithms that are distributed across a Spark cluster. This makes training models against large datasets efficient and fast, and also reduces the time for hypterparameter tuning and model experimentation. 
# MAGIC 
# MAGIC This notebook will walk through the process of building Linear Regression and Random Forest Regression models on time series sensor data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC Most ML models require a certain amount of data engineering work to transform the data attributes into a set of `features` and `labels` that the model can train on. 
# MAGIC 
# MAGIC We will first create a simple linear regression model that predicts the value of `Sensor-Predict` based on the values of all other sensors in the dataset

# COMMAND ----------

# Import the Spark MLlib libraries for feature engineering
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression

# Pull our data into a Spark dataframe
df = spark.sql("select * from sensor_readings")

# Extract the columns that we want in our feature vector
featureColumns = df.drop("timestamp","Sensor-Predict").columns

# First we will use `VectorAssembler` to combine all feature columns into a feature vector (optimized data structure for ML)
assembler = VectorAssembler(inputCols=featureColumns, outputCol="featureVector")
dfVector = assembler.transform(df)

# Then we will scale the values of each sensor to have a standard mean and deviation
scaler = StandardScaler(inputCol="featureVector", outputCol="features", withStd=True, withMean=False)
dfScaled = scaler.fit(dfVector).transform(dfVector)

display(dfScaled.select("features","Sensor-Predict"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training
# MAGIC With our scaled and vectorized feature set, we can now train a linear regression model against the data.
# MAGIC 
# MAGIC Databricks can also visualize model residuals, as well as ROC curves and decision trees.

# COMMAND ----------

# Split the data into a training and test dataset
(trainingData, testingData) = dfScaled.randomSplit([0.7, 0.3])

# Train a linear regression algorith
lr = LinearRegression(featuresCol="features", labelCol="Sensor-Predict")
lrModel = lr.fit(trainingData)

# Plot the residuals
display(lrModel, trainingData)

# COMMAND ----------

# MAGIC %md Let's grab the coefficients, intercept, root mean squared error and R2 for the model

# COMMAND ----------

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark ML Pipelines
# MAGIC Spark Pipelines allow you to build an end to end pipeline for all feature transformations and model training steps. 
# MAGIC 
# MAGIC Let's build a pipeline that contains our vector assembler, standard scaler and linear regression training.

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[assembler, scaler, lr])

(train, test) = df.randomSplit([0.7, 0.3])

# Fit the model
model = pipeline.fit(train)

lrModel = model.stages[-1]

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC Now that we have a model pipeline, we want to tune it and see if we can get the performance (R2, RMSE) up. 
# MAGIC 
# MAGIC MLflow is an open source lifecycle management tool that allows users to train, experiment, package and deploy their models to production. 

# COMMAND ----------

