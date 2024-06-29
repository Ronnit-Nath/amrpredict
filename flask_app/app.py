from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from Bio import SeqIO
import re
import os

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')  # Adjust path as per your directory structure

def load_model(model_type):
    if model_type == 'logistic':
        model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
        vectorizer_path = os.path.join(MODELS_DIR, 'logistic_regression_vectorizer.joblib')
    elif model_type == 'random_forest':
        model_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.joblib')
    else:
        raise ValueError("Invalid model type specified.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_resistance(sequences, model, vectorizer):
    results = []
    for seq in sequences:
        seq = seq.strip().upper()
        
        if len(seq) == 0:
            results.append({"sequence": seq, "prediction": "ERROR: EMPTY SEQUENCE", "probability": 0.0})
            continue
        
        # Check for invalid characters
        invalid_chars = set(seq) - set('ACGTN')
        if invalid_chars:
            results.append({
                "sequence": seq, 
                "prediction": f"ERROR: INVALID CHARACTERS IN DNA SEQUENCE: {', '.join(invalid_chars)}", 
                "probability": 0.0
            })
            continue
        
        try:
            sequence_vectorized = vectorizer.transform([seq])
            prediction = model.predict(sequence_vectorized)
            probability = model.predict_proba(sequence_vectorized)[0][1]
            prediction_label = "Resistant" if prediction[0] == 1 else "Not Resistant"
            results.append({"sequence": seq, "prediction": prediction_label, "probability": float(probability)})
        except Exception as e:
            results.append({
                "sequence": seq, 
                "prediction": f"ERROR: PREDICTION FAILED - {str(e)}", 
                "probability": 0.0
            })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('modelType') or 'logistic'
        input_type = request.form.get('inputType') or 'direct'

        if input_type == 'direct':
            sequences = request.form['sequences'].strip().split(',')
            sequences = [seq.strip() for seq in sequences if seq.strip()]
        elif input_type in ['csv', 'fasta']:
            file = request.files['file']
            if input_type == 'csv':
                df = pd.read_csv(file)
                sequences = df['Sequence'].tolist()
            elif input_type == 'fasta':
                sequences = []
                for record in SeqIO.parse(file, "fasta"):
                    sequences.append(str(record.seq))
        else:
            return jsonify({"error": "Invalid input type."})

        model, vectorizer = load_model(model_type)
        results = predict_resistance(sequences, model, vectorizer)
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
