import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import h5py
from scipy.io import loadmat
import scipy.signal as sg
from scipy.integrate import simps
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import csv

sleep = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
sleep.isna().sum()


sleep.info()

# preprocessing of dataset 

sns.heatmap(sleep.isna(),cmap='BuPu')
sleep.keys()
bp_unique = sleep['Blood Pressure'].unique()
sleep['Blood Pressure']=sleep['Blood Pressure'].apply(lambda x:0 if x in bp_unique else 1)
sleep['Gender']=sleep['Gender'].apply(lambda x:0 if x in ['Female'] else 1)

columns_to_bin = ["Age", "Heart Rate", "Daily Steps", "Sleep Duration", "Physical Activity Level"]

for col in columns_to_bin:
    sleep[col] = pd.cut(sleep[col], bins=4)

from sklearn.preprocessing import LabelEncoder #for converting non-numeric data (String or Boolean) into numbers
LE=LabelEncoder()

categories=['Gender','Age','Occupation','Sleep Duration','Physical Activity Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder']
for label in categories:
    sleep[label]=LE.fit_transform(sleep[label])    

correlation=sleep.corr()
max_6_corr=correlation.nlargest(6,"Sleep Disorder")
sns.heatmap(max_6_corr,annot=True,fmt=".2F",annot_kws={"size":8},linewidths=0.5,cmap='BuPu')
plt.title('Maximum six features affect Sleep Disorder')
plt.show()

x=sleep.iloc[:,:-1]
y=sleep.iloc[:,-1]

x_shape= x.shape
y_shape=y.shape
print('The dimensions of x is : ',x_shape)
print('The dimensions of y is : ',y_shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=32,shuffle=True)

x_train_shape= x_train.shape
x_test_shape= x_test.shape
y_train_shape= y_train.shape
y_test_shape= y_test.shape

print("x train dimensions :",x_train_shape)
print("x test dimensions: ",x_test_shape)
print("y train dimensions :",y_train_shape)
print("y test dimensions :",y_test_shape)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression().fit(x_train,y_train)

LR_training_score= round(LR.score(x_train,y_train)*100,2)
LR_testing_score= round(LR.score(x_test,y_test)*100,2)

print(f"LR training score :",LR_training_score)
print("LR testing score :",LR_testing_score)
LR_y_pred=LR.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier().fit(x_train,y_train)
gbc_model = GradientBoostingClassifier()
gbc_model.fit(x_train, y_train)

GBC_training_score=round(GBC.score(x_train,y_train)*100,2)
GBC_testing_score=round(GBC.score(x_test,y_test)*100,2)

print("GBC training score :",GBC_training_score)
print("GBC testing score :",GBC_testing_score)
GBC_y_pred=gbc_model.predict(x_test)
from sklearn.metrics import confusion_matrix
#looks at sensitivity TD/TP

models_predictions=[LR_y_pred,GBC_y_pred]
model={1:'LR_y_pred',2:'GBC_y_pred'}


plt.figure(figsize=(15,7))
for i,y_pred in enumerate(models_predictions,1) :
    
    cm = confusion_matrix(y_pred,y_test)
    
    plt.subplot(2,3,i)
    sns.heatmap(cm,cmap='BuPu',linewidth=3,fmt='',annot=True,
                xticklabels=['(None)','(Sleep_Apnea)','(Insomnia)'],
                yticklabels=['(None)','(Sleep_Apnea)','(Insomnia)'])
    
    
    plt.title(' CM of  '+ model[i])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)



#ground truth classesa and module predictive class

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.001, 0.01),
    'max_depth': randint(3, 6)
}

# Create the GBC model
gbc_model = GradientBoostingClassifier()

# Create RandomizedSearchCV instance
random_search = RandomizedSearchCV(gbc_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42)

# Fit the model on your training data
random_search.fit(x_train, y_train)

# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.003660699842170359],
    'max_depth': [5]
}

# Create the GBC model
gbc_model = GradientBoostingClassifier()

# Create GridSearchCV instance
grid_search = GridSearchCV(gbc_model, param_grid, cv=3, scoring='accuracy')

# Fit the model on your training data
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
# Create a new GBC model with the best hyperparameters

from sklearn.ensemble import GradientBoostingClassifier

best_gbc_model = GradientBoostingClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth']
)

# gbc_model = GradientBoostingClassifier()
# Fit the best model on your training data
best_gbc_model.fit(x_train, y_train)

# Make predictions using the best model
predictions = best_gbc_model.predict(x_test)
# feature_importances = gbc_model.feature_importances_

print(predictions)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

# Assuming X_train and y_train are your training data
gbc_model = GradientBoostingClassifier()

# Fit the model to your training data
gbc_model.fit(x_train, y_train)

# Now you can use methods and attributes of the model without encountering NotFittedError
predictions = gbc_model.predict(x_test)
feature_importances = gbc_model.feature_importances_

import matplotlib.pyplot as plt

# Sort feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]
sorted_importances = feature_importances[sorted_indices]

# Names of features
feature_names = ['Gender','Age','Occupation','Sleep Duration','Physical Activity Level','BMI Category','Heart Rate','Daily Steps'] + ['Sleep Disorder' + str(i) for i in sorted_indices]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances)
plt.xticks(range(len(feature_names)), feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Plot')
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier  # or your chosen model

model = GradientBoostingClassifier()  # or your chosen model
predictions = cross_val_predict(model, x, y, cv=5)  # 5-fold CV
# df_with_predictions = pd.DataFrame({'Gender': x['Age'], x['Occupation']: x['Sleep Duration'], x['Quality of Sleep'], x['Physical Activity Level', x['BMI Category'],x['Quality of Sleep'], x['Blood Pressure']] 'Predictions': predictions})