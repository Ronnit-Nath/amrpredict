# src/cross_validation.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from data_preparation import prepare_data

def cross_validate_logistic_regression():
    # Prepare data
    file_path = r'../data/genomic_data.csv'
    aro_categories_path = r'../data/aro_categories.tsv'
    aro_index_path = r'../data/aro_index.tsv'
    card_json_path = r'../data/card.json'
    X_train, _, y_train, _ = prepare_data(file_path, aro_categories_path, aro_index_path, card_json_path)

    # Feature extraction
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Cross-validation for Logistic Regression
    logistic_model = LogisticRegression(random_state=42)
    logistic_cv_scores = cross_val_score(logistic_model, X_train_vectorized, y_train, cv=5)

    print("Logistic Regression CV Scores:", logistic_cv_scores)
    print("Logistic Regression CV Mean Score:", logistic_cv_scores.mean())

if __name__ == "__main__":
    cross_validate_logistic_regression()
    print("Logistic Regression cross-validation completed.")
