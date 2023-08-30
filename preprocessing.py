# In this file, we will preprocess the tabular data

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


