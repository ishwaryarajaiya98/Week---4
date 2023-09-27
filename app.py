# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:07:29 2023

@author: Dell
"""

from flask import Flask, render_template, request

# Import necessary libraries for machine learning
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('trained_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    
    if request.method == 'POST':
        # Get the number of study hours input from the user
        study_hours = float(request.form['Hours'])
        
        # Make predictions using the loaded model
        prediction = model.predict([[study_hours]])[0]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

























