import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import preprocessing_of_data
from sklearn.model_selection import train_test_split


df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv') # shape of df is (374, 13) using df.shape

print(df.isna().sum()) # 219 missing values in Sleep Disorder CHECK

X = df.drop(['Sleep Disorder'], axis = 1)
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#X_train.shape = (250, 12) X_test.shape = (124, 12)

X_train.dtypes
# # Person ID                    int64
# Gender                      object
# Age                          int64
# Occupation                  object
# Sleep Duration             float64
# Quality of Sleep             int64
# Physical Activity Level      int64
# Stress Level                 int64
# BMI Category                object
# Blood Pressure              object
# Heart Rate                   int64
# Daily Steps                  int64