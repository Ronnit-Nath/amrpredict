# src/model_training.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preparation import DataPreparation, save_data

def train_model():
    file_paths = [
        '../data/genomic_data.csv', 
        '../data/dataset1/genomic_data_dataset_1.csv', 
        '../data/dataset2/genomic_data_dataset_2.csv',
        '../data/dataset3/genomic_data_dataset_3.csv',
        '../data/genomic_data.fasta', 
        '../data/dataset1/genomic_data_dataset_1.fasta', 
        '../data/dataset2/genomic_data_dataset_2.fasta',
        '../data/dataset3/genomic_data_dataset_3.fasta',
        # Add more file paths as needed for additional data files
    ]
    aro_categories_paths = [
        '../data/aro_categories.tsv',
        '../data/dataset1/aro_categories_dataset_1.tsv',
        '../data/dataset2/aro_categories_dataset_2.tsv',
        '../data/dataset3/aro_categories_dataset_3.tsv',
        # Add more paths for additional ARO categories files
    ]
    aro_index_paths = [
        '../data/aro_index.tsv',
        '../data/dataset1/aro_index_dataset_1.tsv',
        '../data/dataset2/aro_index_dataset_2.tsv',
        '../data/dataset3/aro_index_dataset_3.tsv',
        # Add more paths for additional ARO index files
    ]
    card_json_paths = [
        '../data/card.json',
        '../data/dataset1/card_dataset_1.json',
        '../data/dataset2/card_dataset_2.json',
        '../data/dataset3/card_dataset_3.json',
        # Add more paths for additional CARD JSON files
    ]

    # Initialize DataPreparation instance
    data_prep = DataPreparation(file_paths, aro_categories_paths, aro_index_paths, card_json_paths)
    X_train, X_test, y_train, y_test = data_prep.prepare_data()

    # Feature extraction
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train Logistic Regression model with increased max_iter
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter here
    logistic_model.fit(X_train_vectorized, y_train)

    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_vectorized, y_train)

    # Evaluate models
    y_pred_logistic = logistic_model.predict(X_test_vectorized)
    y_pred_rf = rf_model.predict(X_test_vectorized)

    print("Logistic Regression Model Accuracy:", accuracy_score(y_test, y_pred_logistic))
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_logistic))

    print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Save models and vectorizer
    joblib.dump(logistic_model, '../models/logistic_model.joblib')
    joblib.dump(rf_model, '../models/rf_model.joblib')
    joblib.dump(vectorizer, '../models/vectorizer.joblib')

if __name__ == "__main__":
    train_model()
    print("Model training completed.")
