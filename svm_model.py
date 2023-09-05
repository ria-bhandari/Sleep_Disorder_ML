# Implementing the SVM model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from read_data import preprocessing_of_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Preprocessing of data - encoding categorical values 
df = pd.get_dummies(df, columns = ['Gender', 'Occupation', 'Sleep Disorder'])

# Split data - features and target variable 
x = df.drop(['Person ID', 'BMI Category', 'Blood Pressure'], axis=1)
y = df['BMI Category']

# Train test split of data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(x_train, y_train)

# Predictions 
y_predict = svm_classifier.predict(x_test)

# Evaluation of model
accuracy = accuracy_score(y_test, y_predict)
report = classification_report(y_test, y_predict)

print(f'The accuracy of the SVC model is: {accuracy}')
print(report)

# Creating a confusion matrix

# actual_labels_of_dataset = ['Gender', 'Occupation']
# predicted_labels_of_dataset = ["Sleep Disorder", "No Sleep Disorder"]

# confusion_m = confusion_matrix(actual_labels_of_dataset, predicted_labels_of_dataset)

# print(confusion_m)