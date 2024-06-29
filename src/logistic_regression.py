# src/logistic_regression.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preparation import prepare_data

def train_logistic_regression():
    # Define paths to metadata files
    file_path = r'../data/genomic_data.csv'
    aro_categories_path = r'../data/aro_categories.tsv'
    aro_index_path = r'../data/aro_index.tsv'
    card_json_path = r'../data/card.json'
    
    # Prepare data using specified paths
    X_train, X_test, y_train, y_test = prepare_data(file_path, aro_categories_path, aro_index_path, card_json_path)

    # Feature extraction
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vectorized, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vectorized)
    print("Logistic Regression Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, '../models/logistic_regression_model.joblib')
    joblib.dump(vectorizer, '../models/logistic_regression_vectorizer.joblib')

if __name__ == "__main__":
    train_logistic_regression()
    print("Logistic Regression training completed.")
