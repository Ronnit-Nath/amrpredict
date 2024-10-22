import joblib
import pandas as pd
from Bio import SeqIO
from data_preparation import DataPreparation

def load_saved_model(model_type):
    """
    Load the saved machine learning model and vectorizer based on the model type.

    Parameters:
    - model_type (str): Type of the model ('logistic' or 'random_forest').

    Returns:
    - model: Loaded machine learning model.
    - vectorizer: Loaded feature vectorizer.
    """
    if model_type == 'logistic':
        model = joblib.load('../models/logistic_regression_model.joblib')
        vectorizer = joblib.load('../models/logistic_regression_vectorizer.joblib')
    elif model_type == 'random_forest':
        model = joblib.load('../models/rf_model.joblib')
        vectorizer = joblib.load('../models/vectorizer.joblib')
    else:
        raise ValueError("Invalid model type specified.")
    return model, vectorizer

def predict_resistance(sequence, model, vectorizer, data_preparation_instance):
    """
    Predict if a given DNA sequence is resistant or not based on the loaded model.

    Parameters:
    - sequence (str): DNA sequence to predict resistance for.
    - model: Loaded machine learning model.
    - vectorizer: Loaded feature vectorizer.
    - data_preparation_instance (DataPreparation): Instance of DataPreparation class.

    Returns:
    - tuple: Prediction result and probability ('Resistant', 'Not Resistant', or 'ERROR: NOT A DNA SEQUENCE', probability).
    """
    # Check if the input is a valid DNA sequence using DataPreparation
    if not data_preparation_instance.validate_sequence(sequence):
        return "ERROR: NOT A DNA SEQUENCE", None

    sequence = sequence.upper()  # Convert sequence to uppercase
    sequence_vectorized = vectorizer.transform([sequence])
    prediction_prob = model.predict_proba(sequence_vectorized)[0]

    if model.classes_[1] == 1:  # Check if the model's positive class represents resistance
        resistant_prob = prediction_prob[1]
    else:
        resistant_prob = prediction_prob[0]

    if prediction_prob[1] > prediction_prob[0]:
        return "Resistant", resistant_prob
    else:
        return "Not Resistant", resistant_prob

def main():
    print("**Welcome to AMR Predict**")
    print("AMR Predict uses machine learning models to predict antibiotic resistance based on DNA sequences.")
    print("Our training dataset includes thousands of sequences annotated with resistance labels.")
    print("Currently, we use Logistic Regression and Random Forest models for prediction.")
    
    # Initialize DataPreparation with file paths
    file_paths = [
        '../data/genomic_data.csv', 
        '../data/genomic_data.fasta'
    ]
    aro_categories_paths = [
        '../data/aro_categories.tsv',
    ]
    aro_index_paths = [
        '../data/aro_index.tsv',
    ]
    card_json_paths = [
        '../data/card.json',
    ]
    
    data_prep = DataPreparation(file_paths, aro_categories_paths, aro_index_paths, card_json_paths)
    X_train, X_test, y_train, y_test = data_prep.prepare_data()

    while True:
        print("\nChoose a model:")
        print("1. Logistic Regression")
        print("2. Random Forest")
        print("3. Exit")
        
        model_choice = input("Enter your choice (1, 2, or 3): ")

        if model_choice == '1':
            model_type = 'logistic'
        elif model_choice == '2':
            model_type = 'random_forest'
        elif model_choice == '3':
            print("Exiting program.")
            print("Thank You for using AMR Predict")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        model, vectorizer = load_saved_model(model_type)
        print(f"{model_type.capitalize()} model loaded successfully.")

        while True:
            print("\nChoose input type:")
            print("1. Enter DNA sequences directly")
            print("2. Load sequences from a CSV file")
            print("3. Load sequences from a FASTA file")
            print("4. Back to model selection")
            
            input_choice = input("Enter your choice (1, 2, 3, or 4): ")

            if input_choice == '1':
                print("Enter DNA sequences separated by commas:")
                sequences_input = input("Sequences: ").strip()
                sequences = [seq.strip() for seq in sequences_input.split(',')]
            elif input_choice == '2':
                file_path = input("Enter the CSV file path: ")
                try:
                    df = pd.read_csv(file_path)
                    sequences = df['Sequence'].tolist()
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue
            elif input_choice == '3':
                file_path = input("Enter the FASTA file path: ")
                try:
                    sequences = []
                    for record in SeqIO.parse(file_path, "fasta"):
                        sequences.append(str(record.seq))
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue
            elif input_choice == '4':
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                continue

            results = []
            for idx, sequence in enumerate(sequences, start=1):
                result, probability = predict_resistance(sequence, model, vectorizer, data_prep)
                if probability is not None:
                    results.append(f"{idx}. {sequence}: {result} (Probability: {probability:.2f})")
                else:
                    results.append(f"{idx}. {sequence}: {result}")

            print("\nResults:")
            for result in results:
                print(result)

if __name__ == "__main__":
    main()
