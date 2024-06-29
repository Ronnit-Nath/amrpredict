# src/model_training.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preparation import prepare_data

def train_model():
    # Prepare data
    file_path = r'../data/genomic_data.csv'
    aro_categories_path = r'../data/aro_categories.tsv'
    aro_index_path = r'../data/aro_index.tsv'
    card_json_path = r'../data/card.json'
    X_train, X_test, y_train, y_test = prepare_data(file_path, aro_categories_path, aro_index_path, card_json_path)

    # Feature extraction
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train Logistic Regression model
    logistic_model = LogisticRegression(random_state=42)
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
