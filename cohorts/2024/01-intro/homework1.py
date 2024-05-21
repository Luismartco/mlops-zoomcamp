!pip install pandas openpyxl matplotlib
import pandas as pd
import pickle
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

file_path1 = '/content/drive/MyDrive/yellow_tripdata_2023-01.parquet'
file_path2 = '/content/drive/MyDrive/yellow_tripdata_2023-02.parquet'

df1 = pd.read_parquet(file_path1)
df2 = pd.read_parquet(file_path2)

########################################################################
#1
num_columnas = df1.shape[1]
print(f"El dataset has {num_columnas} columns.")

#Answer:19
######################################################################
#2
# Convert columns to datetime objects
df2.tpep_dropoff_datetime = pd.to_datetime(df2.tpep_dropoff_datetime)
df2.tpep_pickup_datetime = pd.to_datetime(df2.tpep_pickup_datetime)

# Calculate duration in minutes
df2['duration'] = df2.tpep_dropoff_datetime - df2.tpep_pickup_datetime
df2.duration = df2.duration.apply(lambda td: td.total_seconds() / 60)

s = df2.duration.std()
print(s)
#Answer:42.84
###################################################################
#3
L1 = len(df2)

# Filter rows based on duration
df2 = df2[(df2.duration >= 1) & (df2.duration <= 60)]

L2 = len(df2)
r = L2 / L1

print(r)

#Answer:98%
#######################################################################
#4
df2['PU'] = 'PU_' + df2['PULocationID'].astype(str)
df2['DO'] = 'DO_' + df2['DOLocationID'].astype(str)

# Define categorical and numerical columns
categorical = ['PU', 'DO']
numerical = []

# Convert categorical and numerical data into a dictionary format
train_dicts = df2[categorical + numerical].to_dict(orient='records')

# Fit the DictVectorizer to the training data
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

# Define the target variable
target = 'duration'

# Extract the target variable from the DataFrame
y_train = df2[target].values

print(X_train.shape)

#Answer: 515
##############################################################
#5
# Train the LinearRegression model on the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the training data
y_pred = lr.predict(X_train)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_train, y_pred, squared=False)

print(mse)
#Answer:7.778948573497093
##########################################################
#6
# Convert columns to datetime objects
df1.tpep_dropoff_datetime = pd.to_datetime(df1.tpep_dropoff_datetime)
df1.tpep_pickup_datetime = pd.to_datetime(df1.tpep_pickup_datetime)

# Calculate duration in minutes
df1['duration'] = df1.tpep_dropoff_datetime - df1.tpep_pickup_datetime
df1.duration = df1.duration.apply(lambda td: td.total_seconds() / 60)

s = df1.duration.std()
L3 = len(df1)

# Filter rows based on duration
df1 = df1[(df1.duration >= 1) & (df1.duration <= 60)]

L4 = len(df1)
r = L4 / L3
# Define categorical and numerical columns
categorical = ['PU', 'DO']
numerical = []

df1['PU'] = 'PU_' + df1['PULocationID'].astype(str)
df1['DO'] = 'DO_' + df1['DOLocationID'].astype(str)

# Convert categorical and numerical data into a dictionary format
train_dicts = df1[categorical + numerical].to_dict(orient='records')

# Fit the DictVectorizer to the training data
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

# Define the target variable
target = 'duration'

# Extract the target variable from the DataFrame
y_train = df1[target].values

# Train the LinearRegression model on the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the training data
y_pred = lr.predict(X_train)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_train, y_pred, squared=False)

print(mse)

#Answer:7.649261929771859
