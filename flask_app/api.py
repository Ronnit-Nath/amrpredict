from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load('../models/logistic_regression_model.joblib')
vectorizer = joblib.load('../models/logistic_regression_vectorizer.joblib')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json(force=True)
        input_text = data['text']

        # Vectorize the input text
        input_vector = vectorizer.transform([input_text])

        # Make prediction using the model
        prediction = model.predict(input_vector)[0]

        # Prepare response
        response = {
            'prediction': prediction
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})
