# In this file, the sleep apnea data is visualized to determine any outliers or anomalies in the data

import pandas as pd
import matplotlib.pyplot as plt
from read_data import preprocessing_of_data

sleep_apnea_data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Plot 1: Age VS Heart Rate
data = preprocessing_of_data(sleep_apnea_data)
person_age = sleep_apnea_data['Age']
heart_rate = sleep_apnea_data['Heart Rate']
sleep_disorder = data['Sleep Disorder']

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(person_age, heart_rate, c=sleep_disorder, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Sleep Disorder')
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.title('Age vs. Heart Rate')
plt.grid(True)
plt.show()

# Plot 2: Gender VS Sleep Duration
data = preprocessing_of_data(sleep_apnea_data)
gender = data['Gender']
sleep_duration = sleep_apnea_data['Sleep Duration']
sleep_disorder = data['Sleep Disorder']

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(gender, sleep_duration, c=sleep_disorder, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Sleep Disorder')
plt.xlabel('Gender')
plt.ylabel('Sleep Duration')
plt.title('Gender vs. Sleep Duration')
plt.grid(True)
plt.show()