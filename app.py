# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('modelLR.joblib')
vector = joblib.load('vector.joblib')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    userInput = request.form.get('news')
    result = vector.transform([userInput]).toarray()
    # Make prediction
    pred = model.predict(result)
    pred = int(pred[0])
    if pred == 0:
        pred=-1

    return render_template('index.html', label=pred)

if __name__ == "__main__":
    app.run(debug=True)