# Databricks ML Workshop

## Authors:
Samir Gupta

## Overview
This workshop is an introduction to using Databricks for data analysis and machine learning. 

## Data
The following data sources are used in this workshop:
* `sensor_readings.csv` - contains time-series sensor readings in the format <timestamp>|<sensor1>|...|<sensorN>|<sensor-predict>

## Workshop Documentation
See [Databricks ML Workshop.pdf](https://github.com/tomatoTomahto/Databricks_ML_Workshop/blob/master/Databricks%20ML%20Workshop.pdf) for slides on the data science and ML features in Databricks

## Notebooks
The following notebooks should be executed in the following order:
1. `01 - Data Analysis in Spark SQL.py` - using SQL and Python to analyze a dataset in Databricks
2. `02 - Machine Learning Using Spark MLlib.py` - using Spark MLlib to create a machine learning model on time-series data
3. `03 - Model Experimentation with MLflow` - using MLflow to track parameters, metrics, and artifacts in experiments

## Steps
This workshop assumes that you have access to a Databricks environment. 
1. In your workspace, import the [Databricks_ML_Workshop.dbc](https://github.com/tomatoTomahto/Databricks_ML_Workshop/blob/master/Databricks_ML_Workshop.dbc) DBC archive containing all the code
2. Download the `sensor_readings.csv` file and import it as a table into Databricks. `01 - Data Analysis in Spark SQL` walks you through how to do this. 
3. Run the notebooks
