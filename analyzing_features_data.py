# In this file, the features of the dataset will be analyzed to help us in preprocessing

import pandas as pd

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df.head(10).style.set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'lightsteelblue')]}
])

missing_data_in_file = df.isna().sum()
missing_data_df = pd.DataFrame(missing_data_in_file).reset_index()
missing_data_df.columns = ['Column', 'Missing']
print(missing_data_df)

df.describe().style.set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
    {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'lightsteelblue')]}
])
print(df)