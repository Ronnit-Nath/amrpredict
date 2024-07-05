# src/cross_validation.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from data_preparation import DataPreparation
from model_training import train_model  # Import train_model function from model_training.py

def cross_validate_models(file_paths, aro_categories_paths, aro_index_paths, card_json_path):
    # Initialize DataPreparation class
    data_prep = DataPreparation(file_paths, aro_categories_paths, aro_index_paths, card_json_path)

    # Prepare data
    X_train, _, y_train, _ = data_prep.prepare_data()

    # Feature extraction
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Cross-validation for Logistic Regression
    logistic_model = LogisticRegression(random_state=42)
    logistic_cv_scores = cross_val_score(logistic_model, X_train_vectorized, y_train, cv=5)

    print("Logistic Regression CV Scores:", logistic_cv_scores)
    print("Logistic Regression CV Mean Score:", logistic_cv_scores.mean())

    # Cross-validation for Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_cv_scores = cross_val_score(rf_model, X_train_vectorized, y_train, cv=5)

    print("\nRandom Forest CV Scores:", rf_cv_scores)
    print("Random Forest CV Mean Score:", rf_cv_scores.mean())

if __name__ == "__main__":
    file_paths = [
        '../data/genomic_data.csv', 
        '../data/dataset1/genomic_data_dataset_1.csv', 
        '../data/dataset2/genomic_data_dataset_2.csv',
        '../data/dataset3/genomic_data_dataset_3.csv',
        '../data/genomic_data.fasta', 
        '../data/dataset1/genomic_data_dataset_1.fasta', 
        '../data/dataset2/genomic_data_dataset_2.fasta',
        '../data/dataset3/genomic_data_dataset_3.fasta',# Example FASTA file
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
    

    # Perform model training using the train_model function from model_training.py
train_model()

    # Perform cross-validation using the cross_validate_models function
cross_validate_models(file_paths, aro_categories_paths, aro_index_paths, card_json_paths)

print("\nModel training and cross-validation completed.")
