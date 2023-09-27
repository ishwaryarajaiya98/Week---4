# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 22:14:08 2023

@author: Dell
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset (replace 'your_dataset.csv' with the actual filename)
data = pd.read_excel("C:/Users\Dell\.spyder-py3\deploy_model\student_scores.xlsx")

# Split the data into features (study hours) and target (scores)
X = data['Hours'].values.reshape(-1, 1)  # Features
y = data['Scores'].values  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R-squared: {train_score:.2f}")
print(f"Testing R-squared: {test_score:.2f}")

# Save the trained model to a file (e.g., 'trained_model.pkl')
joblib.dump(model, 'trained_model.pkl')
