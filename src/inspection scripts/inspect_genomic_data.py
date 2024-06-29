import pandas as pd

def inspect_genomic_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Columns in {file_path}: {df.columns}")

if __name__ == "__main__":
    file_path = r'../data/genomic_data.csv'
    inspect_genomic_data(file_path)
