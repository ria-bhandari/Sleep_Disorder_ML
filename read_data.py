# In this file, we are reading the sleep apnea dataset from kaggle

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn import SVC

# Load and check the sleep apnea data
#data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')


# Split the data for training and testing
#x = data.drop('Sleep Disorder', axis = 1)
#y = data['Sleep Disorder']


# Split the data into training and testing sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 42)


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df.head(10).style.set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'lightsteelblue')]}
])

missing_data_in_file = df.isna().sum()
missing_data_df = pd.DataFrame(missing_data_in_file).reset_index()
missing_data_df.columns = ['Column', 'Missing']


df.describe().style.set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'lightsteelblue')]}
])

gender_map = {'Male': 0, 'Female': 1} # changing male to 0 and female to 1 in the dataset
df['Gender'] = df['Gender'].map(gender_map)
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal') # replacing all instances of 'Normal Weight' to 'Normal'
bmi_map = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
df['BMI Category'] = df['BMI Category'].map(bmi_map)
sleep_disorder_map = {'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2} # mapping sleep disorders to numbers
df['Sleep Disorder'] = df['Sleep Disorder'].map(sleep_disorder_map)
occupation_map = {'Software Engineer': 1, 
                    'Doctor': 2,
                    'Sales Representative': 3,
                    'Teacher': 4,
                    'Nurse': 5,
                    'Engineer': 6,
                    'Accountant': 7,
                    'Scientist': 8,
                    'Lawyer': 9,
                    'Salesperson': 10,
                    'Manager': 11} # mapping numbers to each occupation
df['Occupation'] = df['Occupation'].map(occupation_map)
for index, row in df.iterrows():
    systolic, diastolic = map(int, row['Blood Pressure'].split('/'))
    if systolic < 120 and diastolic < 80:
        df.at[index, 'Blood Pressure'] = 'Normal'
    elif 120 <= systolic <= 129 and diastolic < 80:
        df.at[index, 'Blood Pressure'] = 'Elevated'
    elif 130 <= systolic <= 139 and diastolic >= 90:
        df.at[index, 'Blood Pressure'] = 'Hypertension Stage 1'
    elif systolic >= 140 and diastolic >= 90:
        df.at[index, 'Blood Pressure'] = 'Hypertension Stage 2'
    elif systolic > 180 and diastolic > 120:
        df.at[index, 'Blood Pressure'] = 'Hypertensive Crisis'

bp_mapping = {
    'Normal': 0,
    'Elevated': 1,
    'Hypertension Stage 1': 2,
    'Hypertension Stage 2': 3,
    'Hypertensive Crisis': 4
}

df['Blood Pressure'] = df['Blood Pressure'].map(bp_mapping)


x = df.drop('Sleep Disorder', axis = 1)
y = df['Sleep Disorder']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 42)



#df['Gender'].value_counts

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(x_train, y_train)
y_predictions = random_forest.predict(x_test)
accuracy = accuracy_score(y_test, y_predictions)
#print(f"Accuracy: {accuracy: .2f}")

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Column: {column}, Unique Values: {unique_values}")