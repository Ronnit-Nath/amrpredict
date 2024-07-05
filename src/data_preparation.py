import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import os
import joblib

class DataPreparation:
    def __init__(self, file_paths, aro_categories_paths, aro_index_paths, card_json_paths):
        self.file_paths = file_paths
        self.aro_categories_paths = aro_categories_paths
        self.aro_index_paths = aro_index_paths
        self.card_json_paths = card_json_paths

    def prepare_data(self):
        # Example implementation, adjust as per your actual data preparation needs
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series()
        y_test = pd.Series()
        return X_train, X_test, y_train, y_test

    def validate_sequence(self, sequence):
        # Basic validation example (adjust as per your requirements)
        valid_bases = set('ATGC')
        if all(base in valid_bases for base in sequence):
            return True
        else:
            return False

    def load_genomic_data(self):
        dfs = []
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            if file_path.endswith('.fasta'):
                records = list(SeqIO.parse(file_path, 'fasta'))
                df = pd.DataFrame([[record.id, str(record.seq)] for record in records], columns=['ID', 'Sequence'])
            else:
                df = pd.read_csv(file_path)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def load_metadata(self):
        # Load ARO categories from multiple files
        aro_categories_dfs = []
        for path in self.aro_categories_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file {path} does not exist.")
            df = pd.read_csv(path, sep='\t')
            # Extract dataset name from path
            dataset_name = self.extract_dataset_name(path)
            if dataset_name:
                df['Dataset'] = dataset_name  # Add dataset name as a column
            aro_categories_dfs.append(df)
        aro_categories = pd.concat(aro_categories_dfs, ignore_index=True)

        # Load ARO indexes from multiple files
        aro_index_dfs = []
        for path in self.aro_index_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file {path} does not exist.")
            df = pd.read_csv(path, sep='\t')
            # Extract dataset name from path
            dataset_name = self.extract_dataset_name(path)
            if dataset_name:
                df['Dataset'] = dataset_name  # Add dataset name as a column
            aro_index_dfs.append(df)
        aro_index = pd.concat(aro_index_dfs, ignore_index=True)

        # Load multiple CARD JSON data
        card_json_dfs = []
        for path in self.card_json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file {path} does not exist.")
            card_json_df = pd.read_json(path)
            card_json_dfs.append(card_json_df)

        return aro_categories, aro_index, card_json_dfs

    def prepare_data(self):
        # Load genomic data and metadata
        df = self.load_genomic_data()
        aro_categories, aro_index, card_jsons = self.load_metadata()

        # Check columns in the data
        print(f"Columns in genomic data: {df.columns}")

        # Assume all sequences in the dataset are resistant
        df['resistant'] = 1

        # Create a non-resistant class for balance
        non_resistant = df.copy()
        non_resistant['Sequence'] = non_resistant['Sequence'].apply(self.generate_random_sequence)
        non_resistant['resistant'] = 0

        # Combine resistant and non-resistant data
        df = pd.concat([df, non_resistant])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['Sequence'], df['resistant'], test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def generate_random_sequence(self, sequence):
        if pd.isna(sequence) or isinstance(sequence, float):
            return ''.join(random.choice('ATGC') for _ in range(50))  # Replace NaN or float with random sequence
        else:
            return ''.join(random.choice('ATGC') for _ in range(len(sequence)))

    def extract_dataset_name(self, file_path):
        # Extract dataset name from file path
        filename = os.path.basename(file_path)
        dataset_name = filename.split('.')[0]  # Adjust based on your file naming convention
        return dataset_name

    def predict_resistance(self, sequence, model_type='logistic'):
        model, vectorizer = load_saved_model(model_type)
        preprocessed_sequence = vectorizer.transform([sequence])
        prediction = model.predict(preprocessed_sequence)
        return prediction

def save_data(X_train, X_test, y_train, y_test, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("Data saved successfully.")

def load_saved_model(model_type):
    models_dir = '../models/'  # Adjust this according to your directory structure
    if model_type == 'logistic':
        model_path = models_dir + 'logistic_regression_model.joblib'
        vectorizer_path = models_dir + 'logistic_regression_vectorizer.joblib'
    elif model_type == 'random_forest':
        model_path = models_dir + 'rf_model.joblib'
        vectorizer_path = models_dir + 'vectorizer.joblib'
    else:
        raise ValueError("Invalid model type specified.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

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
                     
    data_prep = DataPreparation(file_paths, aro_categories_paths, aro_index_paths, card_json_paths)
    X_train, X_test, y_train, y_test = data_prep.prepare_data()

    # Optionally, save the processed data
    output_dir = '../data'
    save_data(X_train, X_test, y_train, y_test, output_dir)

    print("Data preparation completed.")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
