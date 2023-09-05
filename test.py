import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Read the dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Data Preprocessing
# Convert Gender to numerical values (0 for Male, 1 for Female)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# One-hot encode 'BMI Category'
df = pd.get_dummies(df, columns=['BMI Category'])

# Extract relevant features (you can adjust this based on your needs)
X = df[['Age', 'Physical Activity Level', 'Stress Level', 'BMI Category_Normal Weight', 'BMI Category_Overweight', 'BMI Category_Obese']].values

# Convert Sleep Disorder to numerical values (0 for None, 1 for Sleep Apnea, 2 for Insomnia)
df['Sleep Disorder'] = df['Sleep Disorder'].map({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2})
y = df['Sleep Disorder'].values
df.dropna(subset=['y'], inplace=True)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling - normalize data within a particular range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit SVM to training set
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_predict = classifier.predict(X_test)

# Confusion Matrix - how many predicted classes were correctly predicted
confusion_m = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:\n", confusion_m)

accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)

# Visualizing test set results (you can uncomment this part)
# Note: You may need to adapt the visualization for your specific dataset.
# ...

# Example code for scatter plot (Age vs Physical Activity Level)
# plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', label='None')
# plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='green', label='Sleep Apnea')
# plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], c='blue', label='Insomnia')
# plt.xlabel('Age')
# plt.ylabel('Physical Activity Level')
# plt.legend()
# plt.show()
