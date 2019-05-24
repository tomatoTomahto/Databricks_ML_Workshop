# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Lab 1: Analyzing Data in Spark SQL
# MAGIC Spark SQL is a powerful API for processing and analyzing data using familiar SQL and Dataframe frameworks. 
# MAGIC 
# MAGIC In this workshop, we will:
# MAGIC * Upload data to Databricks
# MAGIC * Read it into a Spark dataframe
# MAGIC * Perform simple analysis on it using Spark SQL and Databricks visualizations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Import Data to Databricks
# MAGIC Click on the "Data" tab and upload sensor_readings.csv into Databricks. 
# MAGIC 
# MAGIC Make sure you check the "Header" and "Infer Schema" boxes to allow Databricks to automatically detect header names and data types from the file. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2: Perform SQL Analysis using SparkSQL
# MAGIC Once your table is created, you can analyze the data in it using familiar SQL. Just use the `%sql` magic command to tell Databricks you plan on using SQL in the cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sensor_readings

# COMMAND ----------

# MAGIC %md Let's apply a time series line chart visualization to the data

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT timestamp, `Sensor-1`, `Sensor-3`, `Sensor-5` 
# MAGIC FROM sensor_readings
# MAGIC WHERE timestamp BETWEEN '2019-05-21 06:00:00' AND '2019-05-21 07:00:00'

# COMMAND ----------

# MAGIC %md Suppose we want to understand the distribution of values for a given sensor

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT `Sensor-Predict` FROM sensor_readings

# COMMAND ----------

# MAGIC %md We often want to understand if there is any correlation between multiple variables using a Box Plot

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT `Sensor-1`, `Sensor-3`, `Sensor-5`, `Sensor-7`, `Sensor-Predict` 
# MAGIC FROM sensor_readings
# MAGIC WHERE timestamp BETWEEN '2019-05-21 06:00:00' AND '2019-05-21 07:00:00'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Analysis using PySpark
# MAGIC SQL is great for quick and easy results and visualization. But as Data Scientists, we often want to understand more complex relationships between the data. 
# MAGIC 
# MAGIC Python libraries like PySpark, Numpy and Seaborn allow users to perform more advanced analysis on the data.

# COMMAND ----------

# MAGIC %md Let's compute the correlation matrix of the sensors to see which sensors are more closely correlated to each other

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pull the Spark dataframe into a Pandas Dataframe
df = spark.sql("select * from sensor_readings").toPandas()

sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(9, 7))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                 square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.title.set_text('Correlation Matrix of Sensors')

display(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC Now that we know that there are some clear correlations between `Sensor-Predict` and `Sensors 1, 3, 5 and 7`, let's build an ML model to predict this relationship. 

# COMMAND ----------

