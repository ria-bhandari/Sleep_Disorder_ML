# In this file, we will preprocess the tabular data

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# for visualizing data
# use command "pip install seaborn"
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

# output plot
    # change x="column_name" to change which values the graph dislpays
# sns.set(style="whitegrid")
# sns.histplot(data=df, x="Age", bins=20, kde=True)
# plt.xlabel("Age")
# plt.title("Distribution of Age")
# plt.show()

"""
All columns in csv:
    Person ID,Gender,Age,Occupation,Sleep Duration,Quality of Sleep,Physical Activity Level,Stress Level,BMI Category,Blood Pressure,Heart Rate,Daily Steps,Sleep Disorder

Columns tested for outliers: 
    Age, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, Blood Pressure, Heart Rate, Daily Steps
"""

# Finding upper and lower limit
    # limits are 3 std away from mean
def removeOutliers(df, df_filtered, columnName):
    # defines upper and lower limits
    upper_limit = df[columnName].mean() + 3*df[columnName].std()
    lower_limit = df[columnName].mean() - 3*df[columnName].std()

    # filters out outliers
    df_filtered = df_filtered[(df_filtered[columnName]>lower_limit) & (df_filtered[columnName]<upper_limit)]
    return df_filtered

# Removes outliers in Blood Pressure 
def removeBPOutliers(df, df_filtered):
    df_temp_BP = df.copy()

    # store top and bottom numbers in new DataFrame 'new'
    new = df_temp_BP['Blood Pressure'].str.split('/', n=1, expand=True)

    # add top and bottom numbers to a temp DataFrame to look for outliers
    # and convert strings to ints
    df_temp_BP['Systolic Value'] = new[0]   # top number
    df_temp_BP['Diastolic Value'] = new[1]  # bottom number

    df_temp_BP['Systolic Value'] = df_temp_BP['Systolic Value'].astype(int)
    df_temp_BP['Diastolic Value'] = df_temp_BP['Diastolic Value'].astype(int)

    # Find upper and lower limits of Systolic and Diastolic values
    SV_upper_limit = df_temp_BP['Systolic Value'].mean() + 3*df_temp_BP['Systolic Value'].std()
    SV_lower_limit = df_temp_BP['Diastolic Value'].mean() - 3*df_temp_BP['Diastolic Value'].std()

    DV_upper_limit = df_temp_BP['Diastolic Value'].mean() + 3*df_temp_BP['Diastolic Value'].std()
    DV_lower_limit = df_temp_BP['Diastolic Value'].mean() - 3*df_temp_BP['Diastolic Value'].std()

    # Copies df_filtered DataFrame and adds 'Systolic Value' and 'Diastolic Value' columns
    df_temp_BP_filtered = df_filtered.copy()
    new_filtered = df_temp_BP_filtered['Blood Pressure'].str.split('/', n=1, expand=True)
    df_temp_BP_filtered['Systolic Value'] = new[0]
    df_temp_BP_filtered['Diastolic Value'] = new[1]
    df_temp_BP_filtered['Systolic Value'] = df_temp_BP_filtered['Systolic Value'].astype(int)
    df_temp_BP_filtered['Diastolic Value'] = df_temp_BP_filtered['Diastolic Value'].astype(int)

    # Filter out outliers
    df_temp_BP_filtered = df_temp_BP_filtered[(df_temp_BP_filtered['Systolic Value']>SV_lower_limit) & (df_temp_BP_filtered['Systolic Value']<SV_upper_limit)]
    df_temp_BP_filtered = df_temp_BP_filtered[(df_temp_BP_filtered['Diastolic Value']>DV_lower_limit) & (df_temp_BP_filtered['Diastolic Value']<DV_upper_limit)]

    # Remove 'Systolic Value' and 'Diastolic Value' columns
    df_temp_BP_filtered.drop(columns=['Systolic Value'], inplace=True)
    df_temp_BP_filtered.drop(columns=['Diastolic Value'], inplace=True)

    return df_temp_BP_filtered

# Remove outliers from each testing column defined above
df_filtered = df.copy()
df_filtered = removeOutliers(df, df_filtered, 'Age')
df_filtered = removeOutliers(df, df_filtered, 'Sleep Duration')
df_filtered = removeOutliers(df, df_filtered, 'Quality of Sleep')
df_filtered = removeOutliers(df, df_filtered, 'Physical Activity Level')
df_filtered = removeOutliers(df, df_filtered, 'Stress Level')
df_filtered = removeOutliers(df, df_filtered, 'Heart Rate')
df_filtered = removeOutliers(df, df_filtered, 'Daily Steps')

df_filtered = removeBPOutliers(df, df_filtered)

print(df)
print("---------------------------------------------")
print(df_filtered)

