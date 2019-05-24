# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Lab 3: Model Experimentation with MLflow
# MAGIC MLflow is a powerful, open source framework for model experimentation, packaging, and deployment. 
# MAGIC 
# MAGIC In this notebook, we'll use Databricks Managed MLflow to:
# MAGIC * Run experiments
# MAGIC * Track model performance across those experiments, and 
# MAGIC * Save a pipeline in the MLflow model format

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets
# MAGIC Databricks allows you to build widgets into your notebooks and dashboards that can be referenced in your code

# COMMAND ----------

dbutils.widgets.text("maxIter","100","Max Iterations")
dbutils.widgets.text("regParam","0.0","Regularization Parameter")
dbutils.widgets.text("elasticNetParam","0.0","Elastic Net Param")

# COMMAND ----------

maxIter = int(dbutils.widgets.get("maxIter"))
regParam = float(dbutils.widgets.get("regParam"))
elasticNetParam = float(dbutils.widgets.get("elasticNetParam"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Function to Plot Actuals vs. Predicted
# MAGIC Whenever we re-train a model, we want to visualize how that model performs by plotting Actuals vs. Predicted. This could also plot residuals, or area under the curve, etc.

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_actuals(actuals):
  f, ax = plt.subplots(figsize=(7, 7))
  ax.scatter(actuals["Sensor-Predict"], actuals["prediction"], marker=",")
  ax.set_title("Predictions vs. Actuals")
  ax.set_xlabel("Actuals")
  ax.set_ylabel("Predictions")
  image = f
  
  # Save figure and return
  f.savefig("actual-v-prediction.png")
  plt.close(f)
  return image    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow
# MAGIC The first step in using Mlflow is to configure the tracking UI in Databricks to track your experiment runs and variables within those runs.
# MAGIC 
# MAGIC The code below is exactly the same as the code used in the previous notebook for training a linear regression algorith. We have just put it into a function. 
# MAGIC 
# MAGIC There are at least 3 things that are needed for an experiment:
# MAGIC * `mlflow.start_run` tells MLflow to start tracking the following run as a new experiment
# MAGIC * `mlflow.log_param` tells MLflow to track a particular variable as a *parameter* of the run
# MAGIC * `mlflow.log_metric` tells MLflow to track a particular variable as a *metric* of the run
# MAGIC * `mlflow.<flavor>.log_model` (optional) tells MLflow to log a model including it's dependencies
# MAGIC * `mlflow.log_artifact` (optional) tells MLflow to log a file from local disk (ie. image, config file, dataset, etc.)

# COMMAND ----------

import mlflow
import mlflow.mleap
import mlflow.spark

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# Pull our data into a Spark dataframe
df = spark.sql("select * from sensor_readings")

# Extract the columns that we want in our feature vector
featureColumns = df.drop("timestamp","Sensor-Predict").columns

def trainLRModel(data, maxIter, regParam, elasticNetParam):
  def evalMetrics(summary):
    rmse = summary.rootMeanSquaredError
    r2 = summary.r2
    return (rmse, r2)
  
  with mlflow.start_run() as run:
    # Split our dataset into training and testing
    (train, test) = df.randomSplit([0.7, 0.3])

    # First we will use `VectorAssembler` to combine all feature columns into a feature vector (optimized data structure for ML)
    # Then we will scale the values of each sensor to have a standard mean and deviation
    # Then build a LR model passing in the widget parameters
    # Then build a pipeline with all our stages
    assembler = VectorAssembler(inputCols=featureColumns, outputCol="featureVector")
    scaler = StandardScaler(inputCol="featureVector", outputCol="features", withStd=True, withMean=False)
    lr = LinearRegression(featuresCol="features", labelCol="Sensor-Predict", maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # Fit the model
    model = pipeline.fit(train)

    lrModel = model.stages[-1]
    
    (rmse, r2) = evalMetrics(lrModel.summary)
    
    # Log mlflow parameters to the LR model
    mlflow.log_param("maxIter", maxIter)
    mlflow.log_param("regParam", regParam)
    mlflow.log_metric("elasticNetParam", elasticNetParam)
    
    # Log mlflow metrics for the LR model
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    
    # Log model generated
    mlflow.spark.log_model(model, "lrModel")
    
    # Log the actuals vs. predicted plot
    predictions = model.transform(test).toPandas()
    image = plot_actuals(predictions)
    mlflow.log_artifact("actual-v-prediction.png")

    return run.info

# COMMAND ----------

trainLRModel(df, maxIter, regParam, elasticNetParam)