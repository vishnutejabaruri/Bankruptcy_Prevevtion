from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)


model = joblib.load("model.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':  

        features = [
            float(request.form['industrial_risk']),
            float(request.form['management_risk']),
            float(request.form['financial_flexibility']),
            float(request.form['credibility']),
            float(request.form['competitiveness']),
            float(request.form['operating_risk']),
        ]
        
        
        data = {
            "industrial_risk":[features[0]],
            "financial_flexibility":[features[2]],
            "credibility":[features[3]],
            "competitiveness":[features[4]]
        }
        
        X = pd.DataFrame(data)
        prediction = model.predict(X)
        result = encoder.inverse_transform(prediction)[0]

        return render_template('result.html', features=features, prediction=result)

@app.route('/new_prediction')
def new_prediction():
    return redirect(url_for('index'))

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    app.run(port=8000)

