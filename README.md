# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection and Preprocessing: Load the weather station dataset and drop irrelevant columns (time, bat). Handle missing values using dropna() and encode the categorical column wind_direction using LabelEncoder to convert it into numerical format.
2. Feature Engineering and Data Splitting: Create the target variable 'rain' by classifying humidity greater than 80 as rainy (1) and otherwise as not rainy (0). Separate the dataset into independent variables (X) and dependent variable (y), then split into training and testing sets using an 80:20 ratio.
3. Model Training: Initialize the Random Forest Classifier with 100 decision trees (n_estimators=100) and a defined random state. Fit the model on the training data (X_train, y_train) where each tree learns different patterns from random subsets of the data.
4. Model Evaluation and Visualization: Predict the output for test data and evaluate using Accuracy Score, Confusion Matrix, and Classification Report. Visualize the feature importance scores as a bar chart to identify which weather parameters contribute most to the prediction.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: SUDESAN T
RegisterNumber: 212225240161
*/
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

uploaded = files.upload()

df = pd.read_csv('weather-station-eee-block_2024_07_13.csv')

df.drop(columns=['time', 'bat'], inplace=True)
df.dropna(inplace=True)

le = LabelEncoder()
df['wind_direction'] = le.fit_transform(df['wind_direction'])

df['rain'] = (df['hum'] > 80).astype(int)

X = df.drop('rain', axis=1)
y = df['rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Plot
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), importances[indices], color='green', alpha=0.7, edgecolor='darkgreen')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right', fontsize=11)
plt.xlabel('Features', fontsize=13)
plt.ylabel('Importance Score', fontsize=13)
plt.title('Random Forest - Feature Importance for Weather Prediction', fontsize=13)
plt.tight_layout()
plt.show()
```

## Output:

<img width="927" height="773" alt="{F41C46EB-5734-403C-8A24-3E98B64AAC1E}" src="https://github.com/user-attachments/assets/1df954f0-e9af-4b20-b2cc-4d7488136c47" />


## Result:
The model achieved 100% accuracy — humidity being the dominant feature makes rain prediction perfectly separable in this dataset.
