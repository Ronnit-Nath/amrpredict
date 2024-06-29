import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

def prepare_data(file_path, aro_categories_path, aro_index_path, card_json_path):
    # Load the genomic data
    df = pd.read_csv(file_path)

    # Load metadata
    aro_categories = pd.read_csv(aro_categories_path, sep='\t')
    aro_index = pd.read_csv(aro_index_path, sep='\t')
    card_json = pd.read_json(card_json_path)

    # Check columns in the data
    print(f"Columns in {file_path}: {df.columns}")

    # Merge with aro_categories
    df = df.merge(aro_categories, left_on='Description', right_on='ARO Name', how='left')

    # Merge with aro_index
    df = df.merge(aro_index, left_on='Description', right_on='Protein Accession', how='left')

    # Assume all sequences in the dataset are resistant
    df['resistant'] = 1

    # Create a non-resistant class for balance
    non_resistant = df.copy()
    non_resistant['Sequence'] = non_resistant['Sequence'].apply(lambda x: ''.join(random.choice('ATGC') for _ in range(len(x))))
    non_resistant['resistant'] = 0

    # Combine resistant and non-resistant data
    df = pd.concat([df, non_resistant])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['Sequence'], df['resistant'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def load_model(model_type):
    if model_type == 'logistic_regression':
        model = joblib.load('../models/logistic_regression_model.joblib')
        vectorizer = joblib.load('../models/logistic_regression_vectorizer.joblib')
    elif model_type == 'random_forest':
        model = joblib.load('../models/rf_model.joblib')
        vectorizer = joblib.load('../models/vectorizer.joblib')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, vectorizer

if __name__ == "__main__":
    file_path = r'../data/genomic_data.csv'
    aro_categories_path = r'../data/aro_categories.tsv'
    aro_index_path = r'../data/aro_index.tsv'
    card_json_path = r'../data/card.json'
    X_train, X_test, y_train, y_test = prepare_data(file_path, aro_categories_path, aro_index_path, card_json_path)
    print("Data preparation completed.")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
